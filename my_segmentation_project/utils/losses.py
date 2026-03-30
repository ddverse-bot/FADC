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



file_path = "medical_segmentation_project/utils/losses.py"


os.makedirs(os.path.dirname(file_path), exist_ok=True)


with open(file_path, "w") as f:
    f.write(losses_content.strip())


print(f"Generated {file_path} with loss function definitions.")