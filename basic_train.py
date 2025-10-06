from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from torchvision.models import vit_b_16
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

N_CLASSES = 19168

dst_path = "/media/disk/vista/BBDD_video_image/ImageNet21kOfficial/winter21_whole"


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

print(f"Total Params: {get_n_params(model)}")

train_transforms = transforms.Compose([
    ## Preprocessing
    transforms.Resize(256),
    transforms.CenterCrop(224),
    ## Augmentations
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandAugment(num_ops=2, magnitude=10),

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

BATCH_SIZE = 512

dst_rdy = DataLoader(dst,
                     batch_size=BATCH_SIZE,
                     shuffle=True,
                     num_workers=16,
                     pin_memory=True)
N_BATCHES = len(dst_rdy)
print(f"DataLoader length: {N_BATCHES}")

for batch in dst_rdy:
    break
x, y = batch
print(f"X: {x.shape} Y: {y.shape}")

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.03, decoupled_weight_decay=True)

## Schedulers
EPOCHS = 1
LINEAR_STEPS = 10000
scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                               start_factor=1/10,
                                               end_factor=1.,
                                               total_iters=LINEAR_STEPS)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=N_BATCHES*EPOCHS-LINEAR_STEPS)
scheduler_final = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                        schedulers=[scheduler1, scheduler2],
                                                        milestones=[LINEAR_STEPS])

## Training Loop
model.train()
epochs_loss = []
step = 0
for epoch in range(EPOCHS):
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
        batch_loss.append(loss.item())
        step += 1

        scheduler_final.step()
        batch_end = time()
        if step % 100 == 0 or step == 1:
            print(f"Step {step} ({batch_end-batch_start:.3f}s/batch) --> Loss: {np.mean(batch_loss).item()}")

    torch.save(model.state_dict(), f"vit-im21k-{step}.pth")
    epochs_loss.append(np.mean(batch_loss).item())
    # break
