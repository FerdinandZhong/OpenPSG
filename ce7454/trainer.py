import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
# from apex import amp
from time import sleep, time
from torch.utils.tensorboard import SummaryWriter


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class BaseTrainer:
    def __init__(
        self,
        net: nn.Module,
        train_loader: DataLoader,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        epochs: int = 100,
        alpha: float = 0,
        is_fp16: bool = False,
        log_rate: int = 50,
        log_dir: str = "./outputs"
    ) -> None:
        self.is_fp16 = is_fp16
        self.net = net
        self.optimizer = torch.optim.SGD(
            net.parameters(),
            learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                epochs * len(train_loader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / learning_rate,
            ),
        )

        self.scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=is_fp16)
        #     self.net, self.optimizer = amp.initialize(
        #         models=net.cuda(), optimizers=self.optimizer, opt_level="O1"
        #     )
        #     amp._amp_state.loss_scalers[0]._loss_scale = 2**20
        # else:
        #     self.net = net.cuda()

        self.train_loader = train_loader

        self.alpha = alpha
        self.log_rate = log_rate
        self.total_steps = 0
        self.tensorboard_writter = SummaryWriter(log_dir)
        self.start_time = time()

    def _compute_kl_loss(self, logits):
        p,  q = torch.split(logits, logits.size(0)//2, dim=0)
        p_loss = F.kl_div(
            F.log_softmax(p), F.softmax(q), reduction="none"
        )
        q_loss = F.kl_div(
            F.log_softmax(q), F.softmax(p), reduction="none"
        )

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = p_loss + q_loss
        return loss

    def forward(self, data, target):
        logits = self.net(data)
        return F.binary_cross_entropy_with_logits(logits, target, reduction="sum")

    def forward_reg(self, data, target):
        # data_concat = torch.cat([data, data.clone()], 0)
        logits_1 = self.net(data)
        logits_2 = self.net(data)
        logits = torch.cat([logits_1, logits_2], dim=0)
        target = torch.cat([target, target.clone()])
        loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction="sum"
        )
        loss += self.alpha * self._compute_kl_loss(logits)

        return loss

    def train_epoch(self):
        self.net.train()  # enter train mode

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        with tqdm(range(1, len(train_dataiter) + 1)) as pbar:
            for _ in range(1, len(train_dataiter) + 1):
                self.total_steps += 1
                # for train_step in tqdm(range(1, 5)):
                data, soft_label = next(train_dataiter)
                data = data.cuda()
                target = soft_label.cuda()
                # forward
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.is_fp16):
                    if self.alpha > 0:
                        loss = self.forward_reg(data, target)
                    else:
                        loss = self.forward(data, target)

                # backward
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # if self.is_fp16:
                #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                #     loss.backward()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # exponential moving average, show smooth values
                with torch.no_grad():
                    loss_avg = loss_avg * 0.8 + float(loss) * 0.2

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Last_loss": f"{loss:.2f}",
                    }
                )
                if self.total_steps % self.log_rate == 0:
                    self.tensorboard_writter.add_scalar(
                        "training loss (steps)", loss_avg, self.total_steps
                    )
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.tensorboard_writter.add_scalar(
                        "learning rate (steps)", current_lr, self.total_steps
                    )
                    used_time = time() - self.start_time
                    self.tensorboard_writter.add_scalar(
                        "training loss (seconds)", loss_avg, round(used_time*1000, 2)
                    )


        metrics = {}
        metrics["train_loss"] = loss_avg

        return metrics
