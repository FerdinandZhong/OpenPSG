import argparse
import os
import time

import torch
from dataset import PSGClsDataset
from evaluator import Evaluator
from torch.utils.data import DataLoader
from torchvision.models import (
    resnet50,
    regnet_y_32gf,
    efficientnet_v2_l,
    ResNet50_Weights,
    RegNet_Y_32GF_Weights,
    EfficientNet_V2_L_Weights
)
from trainer import BaseTrainer, cosine_annealing
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from tqdm import tqdm
import torch.nn.functional as F
from functools import partial



parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="resnet50")
parser.add_argument("--epoch", type=int, default=36)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0.0005)
parser.add_argument("--pretrained", type=str, default="true")
parser.add_argument("--is_fp16", type=str, default="false")
parser.add_argument("--alpha", type=float, default=0)
parser.add_argument("--gpu_device", type=int, default=0)

args = parser.parse_args()
is_fp16 = args.is_fp16 == "true"
pretrained = args.pretrained == "true"
device = torch.device(f"cuda:{args.gpu_device}")

backbone_model_dict = {
    "regnet": regnet_y_32gf,
    "resnet50": resnet50,
    "efficientnet": efficientnet_v2_l,
}

model_weights = {
    "regnet": RegNet_Y_32GF_Weights,
    "resnet50": ResNet50_Weights,
    "efficientnet": EfficientNet_V2_L_Weights,
}


savename = f"{args.model_name}_e{args.epoch}_lr{args.lr}_bs{args.batch_size}_m{args.momentum}_wd{args.weight_decay}_alpha{args.alpha}"
os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# loading dataset
train_dataset = PSGClsDataset(stage="train", is_fp16=is_fp16)
train_dataloader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
)

val_dataset = PSGClsDataset(stage="val", is_fp16=is_fp16)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

test_dataset = PSGClsDataset(stage="test", is_fp16=is_fp16)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)


print("Data Loaded...", flush=True)

# loading model
# model = resnet50(pretrained=True)
if pretrained:
    model = backbone_model_dict[args.model_name](
        weights=model_weights[args.model_name]
    )
    if args.model_name == "efficientnet":
        current_classifier = model.classifier
        model.classifier = torch.nn.Sequential(
            current_classifier[0],
            torch.nn.Linear(current_classifier[1].in_features, 56),
        )
    else:
        current_fc = model.fc
        model.fc = torch.nn.Linear(current_fc.in_features, 56)
else:
    model = backbone_model_dict[args.model_name](num_classes=56)
model.to(device)

print("Model Loaded...", flush=True)

# loading trainer
trainer = BaseTrainer(
    model,
    train_dataloader,
    learning_rate=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    epochs=args.epoch,
    alpha=args.alpha,
    is_fp16=is_fp16,
)
evaluator = Evaluator(model, k=3, is_fp16=is_fp16)

def train_cifar(config, train_loader, val_dataloader, net, epochs, checkpoint_dir=None):
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.cuda()
    optimizer = torch.optim.SGD(
        net.parameters(),
        config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / config["lr"],
        ),
    )

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(0, epochs):
        loss_avg = 0.0
        train_dataiter = iter(train_loader)
        with tqdm(range(1, len(train_dataiter) + 1)) as pbar:
            for _ in range(1, len(train_dataiter) + 1):
                batch = next(train_dataiter)
                data = batch[0].cuda()
                target = batch[1].cuda()
                # forward
                logits = net(data)
                loss = F.binary_cross_entropy_with_logits(logits,
                                                        target,
                                                        reduction='sum')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    loss_avg = loss_avg * 0.8 + float(loss) * 0.2
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Last_loss": f"{loss:.2f}",
                    }
                )
        val_metrics = evaluator.eval_recall(val_dataloader)
        print(
            "Epoch {:3d} | Train Loss {:.4f} | Test Loss {:.3f} | mR {:.2f}".format(
                (epoch + 1),
                loss_avg,
                val_metrics["test_loss"],
                100.0 * val_metrics["mean_recall"],
            ),
            flush=True,
        )

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "check")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
        
        tune.report(train_loss = loss_avg, val_loss = val_metrics["test_loss"], val_accuracy = 100.0 * val_metrics["mean_recall"])
    print("finish tuning")

gpus_per_trial = 2

config = {
    "momentum": tune.choice([0.5, 0.7, 0.9]),
    "weight_decay": tune.loguniform(1e-5, 1e-3),
    "lr": tune.loguniform(1e-4, 1e-1)
}

scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=10,
    grace_period=1,
    reduction_factor=2)
reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["loss", "training_iteration"])

result = tune.run(
    partial(train_cifar, train_loader=train_dataloader, val_dataloader=val_dataloader, net=model, epochs=args.epoch),
    resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
    config=config,
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_score_attr="val_accuracy",
    keep_checkpoints_num=1
)

# tuner = tune.Tuner.restore(path="/root/ray_results/train_cifar_2022-10-10_21-05-59", resume_errored=True)
# result = tuner.fit()

best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(
    best_trial.last_result["accuracy"]))

best_checkpoint_dir = best_trial.checkpoint.value
model_state, optimizer_state = torch.load(os.path.join(
    best_checkpoint_dir, "checkpoint"))
model.load_state_dict(model_state)
test_evaluator = Evaluator(model, k=3)
check_metrics = test_evaluator.eval_recall(val_dataloader)
print(check_metrics["mean_recall"])

result = test_evaluator.submit(test_dataloader)

# save into the file
with open(f"results/tuned_{args.model_name}.txt", "w") as writer:
    for label_list in result:
        a = [str(x) for x in label_list]
        save_str = " ".join(a)
        writer.writelines(save_str + "\n")
print("Result Saved!", flush=True)