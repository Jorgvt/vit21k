import os
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from torchvision.models import vit_b_16
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

import torchmetrics
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

N_CLASSES = 19168

dst_path = "/media/disk/vista/BBDD_video_image/ImageNet21kOfficial/winter21_whole"

config = {
        "RESIZE": 256,
        "CENTER_CROP": 224,
        "PROB_HOR_FLIP": 0.5,
        "RANDAUGMENT_OPS": 2,
        "RANDAUGMENT_MAG": 10,
        "BATCH_SIZE": 512,
        "LEARNING_RATE": 1e-3,
        "BETA1": 0.9,
        "BETA2": 0.999,
        "WEIGHT_DECAY": 0.03,
        "DECOUPLED_WEIGHT_DECAY": True,
        "EPOCHS": 1,
        "LINEAR_STEPS": 10000,
        "START_FACTOR": 1/10,
        "END_FACTOR": 1.,
        }

wandb.init(
        project="vit-21k",
        name="vit-b-16",
        job_type="training",
        config=config,
        mode="online",
        )
config = wandb.config

## Metric definition
loss_metric = torchmetrics.aggregation.MeanMetric().to(device)
acc_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=N_CLASSES).to(device)

model = vit_b_16(weights=None, num_classes=N_CLASSES)
model.to(device)

print(f"GPU memory consumed after loading: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

n_params = get_n_params(model)
wandb.run.summary.update({"total_parameters": n_params})
print(f"Total Params: {n_params}")

train_transforms = transforms.Compose([
    ## Preprocessing
    transforms.Resize(config.RESIZE),
    transforms.CenterCrop(config.CENTER_CROP),
    ## Augmentations
    transforms.RandomHorizontalFlip(config.PROB_HOR_FLIP),
    transforms.RandAugment(num_ops=config.RANDAUGMENT_OPS, magnitude=config.RANDAUGMENT_MAG), 
    ## Preprocessing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

## based on light2
# train_augmentations = transforms.Compose([
#     # transforms.v2.MixUp(alpha=0.2),
#     transforms.RandAugment(num_ops=2, magnitude=10)
# ])

# def collate_fn(batch):
#     return cutmix_or_mixup(*default_collate(batch))

dst = ImageFolder(dst_path,
                  transform=train_transforms)
print(f"Dataset length: {len(dst)}")


dst_rdy = DataLoader(dst,
                     batch_size=config.BATCH_SIZE,
                     shuffle=True,
                     num_workers=16,
                     pin_memory=True)
N_BATCHES = len(dst_rdy)
wandb.run.summary.update({"N_BATCHES": N_BATCHES})
print(f"DataLoader length: {N_BATCHES}")

for batch in dst_rdy:
    break
x, y = batch
print(f"X: {x.shape} Y: {y.shape}")

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2), weight_decay=config.WEIGHT_DECAY, decoupled_weight_decay=config.DECOUPLED_WEIGHT_DECAY)

## Schedulers
scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                               start_factor=config.START_FACTOR,
                                               end_factor=config.END_FACTOR,
                                               total_iters=config.LINEAR_STEPS)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=N_BATCHES*config.EPOCHS-config.LINEAR_STEPS)
scheduler_final = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                        schedulers=[scheduler1, scheduler2],
                                                        milestones=[config.LINEAR_STEPS])

## Training Loop
model.train()
epochs_loss, epochs_acc = [], []
step = 0
for epoch in range(config.EPOCHS):
    batch_loss = []
    for batch in dst_rdy:
        batch_start = time()
        optimizer.zero_grad()

        x, y = batch
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()

        ## Gradient Clipping
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.)

        optimizer.step()
        # batch_loss.append(loss.item())

        ## Metric
        loss_metric.update(loss)
        acc = acc_metric(pred, y)
        step += 1

        scheduler_final.step()
        batch_end = time()
        if step % 100 == 0 or step == 1:
            batch_time = batch_end-batch_start
            loss_, acc_ = loss_metric.compute(), acc_metric.compute()
            print(f"Step {step} ({batch_time:.3f}s/batch) --> Loss: {loss_} | Acc: {acc_}")
            wandb.log({"step": step,
                       "train_loss": loss_,
                       "train_accuracy": acc_,
                       "learning_rate": scheduler_final.get_last_lr()[0],
                       "batch_time": batch_time,
                       })

    torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"vit-im21k-{step}.pth"))
    torch.save(optimizer.state_dict(), os.path.join(wandb.run.dir, f"vit-im21k-optimizer-{step}.pth"))
    torch.save(scheduler_final.state_dict(), os.path.join(wandb.run.dir, f"vit-im21k-scheduler-{step}.pth"))
    epochs_loss.append(loss_metric.compute())
    epochs_acc.append(acc_metric.compute())

    ## Reset metrics
    loss_metric.reset()
    acc_metric.reset()
    # break

wandb.finish()
