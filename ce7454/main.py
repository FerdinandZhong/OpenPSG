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
from trainer import BaseTrainer
from torch.utils.tensorboard import SummaryWriter

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
parser.add_argument("--log_rate", type=int, default=50)
parser.add_argument("--log_dir", type=str, default="./outputs")
parser.add_argument("--two_stage", action='store_true')


args = parser.parse_args()
is_fp16 = args.is_fp16 == "true"
pretrained = args.pretrained == "true"

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

savename = f"{savename}_twostage" if args.two_stage else savename

os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./results", exist_ok=True)

print(f"two stage: {args.two_stage}")

# loading dataset
train_dataset = PSGClsDataset(stage="train", is_fp16=is_fp16, two_stage=args.two_stage)
train_dataloader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
)

val_dataset = PSGClsDataset(stage="val", is_fp16=is_fp16, two_stage=args.two_stage)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8)

test_dataset = PSGClsDataset(stage="test", is_fp16=is_fp16, two_stage=args.two_stage)
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

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model.cuda()

print("Model Loaded...", flush=True)
tensorboard_writter = SummaryWriter(args.log_dir)

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
    log_rate=args.log_rate,
    log_dir=args.log_dir
)
evaluator = Evaluator(model, k=3, is_fp16=is_fp16)

# train!
print("Start Training...", flush=True)
begin_epoch = time.time()
best_val_recall = 0.0
for epoch in range(0, args.epoch):
    train_metrics = trainer.train_epoch()
    val_metrics = evaluator.eval_recall(val_dataloader)

    # show log
    print(
        "{} | Epoch {:3d} | Time {:5d}s | Train Loss {:.4f} | Test Loss {:.3f} | mR {:.2f}".format(
            savename,
            (epoch + 1),
            int(time.time() - begin_epoch),
            train_metrics["train_loss"],
            val_metrics["test_loss"],
            100.0 * val_metrics["mean_recall"],
        ),
        flush=True,
    )

    tensorboard_writter.add_scalars(f'loss (epoch)', {
        'training': train_metrics["train_loss"],
        'validation': val_metrics["test_loss"],
    }, epoch+1)
    
    tensorboard_writter.add_scalar(
        "validation recall (epoch)", 100.0 * val_metrics["mean_recall"], epoch+1
    )

    # save model
    if val_metrics["mean_recall"] >= best_val_recall:
        torch.save(
            model.state_dict(), f"./checkpoints/{args.model_name}/1011/{savename}_best.ckpt"
        )
        best_val_recall = val_metrics["mean_recall"]

print("Training Completed...", flush=True)

# saving result!
print("Loading Best Ckpt...", flush=True)
checkpoint = torch.load(f"checkpoints/{args.model_name}/1011/{savename}_best.ckpt")
model.load_state_dict(checkpoint)
test_evaluator = Evaluator(model, k=3)
check_metrics = test_evaluator.eval_recall(val_dataloader)
print(check_metrics["mean_recall"])
# best_val_recall = check_metrics["mean_recall"]
if best_val_recall == check_metrics["mean_recall"]:
    print(
        "Successfully load best checkpoint with acc {:.2f}".format(
            100 * best_val_recall
        ),
        flush=True,
    )
else:
    print("Fail to load best checkpoint")
result = test_evaluator.submit(test_dataloader)

# save into the file
with open(f"results/{savename}_{best_val_recall}.txt", "w") as writer:
    for label_list in result:
        a = [str(x) for x in label_list]
        save_str = " ".join(a)
        writer.writelines(save_str + "\n")
print("Result Saved!", flush=True)

