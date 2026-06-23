import os
import torch
import torch.nn as nn

from tqdm import tqdm

from utils.data_preprocessing import prepare_busi_dataset

from utils.losses import (
CombinedLoss,
SegmentationCalibrationLoss
)

from models import (
UNet,
UNetFADC,
UNetPlusPlus,
UNetPlusPlusFADC,
AttentionUNet,
AttentionUNetFADC,
TransUNet,
TransUNetFADC
)

# --------------------------------------------------

# CONFIG

# --------------------------------------------------

MODEL_NAME = "UNetFADC"

USE_CALIBRATION = True

EPOCHS = 50

LR = 1e-4

DEVICE = (
"cuda"
if torch.cuda.is_available()
else "cpu"
)

CHECKPOINT_DIR = "checkpoints"

os.makedirs(
CHECKPOINT_DIR,
exist_ok=True
)

# --------------------------------------------------

# MODEL FACTORY

# --------------------------------------------------

def build_model(name):

```
models = {

    "UNet":
        UNet(),

    "UNetFADC":
        UNetFADC(),

    "UNetPlusPlus":
        UNetPlusPlus(),

    "UNetPlusPlusFADC":
        UNetPlusPlusFADC(),

    "AttentionUNet":
        AttentionUNet(),

    "AttentionUNetFADC":
        AttentionUNetFADC(),

    "TransUNet":
        TransUNet(),

    "TransUNetFADC":
        TransUNetFADC()
}

return models[name]
```

# --------------------------------------------------

# DICE SCORE

# --------------------------------------------------

def dice_score(
preds,
targets,
smooth=1e-6
):

```
preds = preds.view(-1)

targets = targets.view(-1)

intersection = (
    preds * targets
).sum()

dice = (
    2.0 * intersection + smooth
) / (
    preds.sum()
    + targets.sum()
    + smooth
)

return dice.item()
```

# --------------------------------------------------

# TRAIN LOOP

# --------------------------------------------------

def train_epoch(
model,
loader,
optimizer,
criterion
):

```
model.train()

running_loss = 0

for images, masks in tqdm(loader):

    images = images.to(DEVICE)

    masks = masks.to(DEVICE)

    optimizer.zero_grad()

    logits = model(images)

    if USE_CALIBRATION:

        loss, seg_loss, cal_loss = criterion(
            logits,
            masks
        )

    else:

        loss = criterion(
            logits,
            masks
        )

    loss.backward()

    optimizer.step()

    running_loss += loss.item()

return (
    running_loss
    /
    len(loader)
)
```

# --------------------------------------------------

# VALIDATION

# --------------------------------------------------

def validate(
model,
loader,
criterion
):

```
model.eval()

val_loss = 0

dice_list = []

with torch.no_grad():

    for images, masks in loader:

        images = images.to(
            DEVICE
        )

        masks = masks.to(
            DEVICE
        )

        logits = model(images)

        if USE_CALIBRATION:

            loss, _, _ = criterion(
                logits,
                masks
            )

        else:

            loss = criterion(
                logits,
                masks
            )

        probs = torch.sigmoid(
            logits
        )

        preds = (
            probs > 0.5
        ).float()

        dice = dice_score(
            preds,
            masks
        )

        dice_list.append(
            dice
        )

        val_loss += loss.item()

return (

    val_loss
    /
    len(loader),

    sum(dice_list)
    /
    len(dice_list)
)
```

# --------------------------------------------------

# MAIN

# --------------------------------------------------

def main():

```
train_loader, val_loader = (
    prepare_busi_dataset()
)

model = build_model(
    MODEL_NAME
).to(DEVICE)

if USE_CALIBRATION:

    criterion = (
        SegmentationCalibrationLoss(
            calibration_weight=0.1
        )
    )

else:

    criterion = (
        CombinedLoss()
    )

optimizer = (
    torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5
    )
)

scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )
)

best_dice = 0

for epoch in range(EPOCHS):

    train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )

    val_loss, val_dice = validate(
        model,
        val_loader,
        criterion
    )

    scheduler.step()

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Dice: {val_dice:.4f}"
    )

    if val_dice > best_dice:

        best_dice = val_dice

        torch.save(

            model.state_dict(),

            os.path.join(
                CHECKPOINT_DIR,
                f"{MODEL_NAME}_best.pth"
            )
        )

        print(
            "Best model saved."
        )

print(
    f"Best Dice: {best_dice:.4f}"
)
```

if **name** == "**main**":

```
main()
```
