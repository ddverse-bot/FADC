import os
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss() # Often preferred for binary segmentation over BCE
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        # BCEWithLogitsLoss expects raw logits, not probabilities
        bce = self.bce_loss(inputs, targets)

        # For DiceLoss, we need probabilities, so apply sigmoid to inputs
        inputs_sigmoid = torch.sigmoid(inputs)
        dice = self.dice_loss(inputs_sigmoid, targets)

        return self.bce_weight * bce + self.dice_weight * dice
class ML1ACE(nn.Module):

def __init__(
    self,
    n_bins=15
):
    super().__init__()

    self.n_bins = n_bins

def forward(
    self,
    probs,
    targets
):

    probs = probs.view(-1)

    targets = targets.view(-1)

    bin_boundaries = torch.linspace(
        0,
        1,
        self.n_bins + 1,
        device=probs.device
    )

    ace = 0.0

    for i in range(self.n_bins):

        lower = bin_boundaries[i]

        upper = bin_boundaries[i + 1]

        mask = (
            (probs >= lower)
            &
            (probs < upper)
        )

        if mask.sum() > 0:

            conf = probs[mask].mean()

            acc = targets[mask].float().mean()

            ace += torch.abs(
                conf - acc
            )

    return ace / self.n_bins
--------------------------------------------------
Combined Segmentation + Calibration
--------------------------------------------------

class SegmentationCalibrationLoss(nn.Module):

def __init__(
    self,
    calibration_weight=0.1
):
    super().__init__()

    self.seg_loss = CombinedLoss()

    self.cal_loss = ML1ACE()

    self.calibration_weight = (
        calibration_weight
    )

def forward(
    self,
    logits,
    targets
):

    seg = self.seg_loss(
        logits,
        targets
    )

    probs = torch.sigmoid(logits)

    cal = self.cal_loss(
        probs,
        targets
    )

    total = (
        seg
        +
        self.calibration_weight * cal
    )

    return total, seg, cal


file_path = "medical_segmentation_project/utils/losses.py"


os.makedirs(os.path.dirname(file_path), exist_ok=True)


with open(file_path, "w") as f:
    f.write(losses_content.strip())


print(f"Generated {file_path} with loss function definitions.")
