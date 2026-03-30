import os
import torch
import numpy as np
from sklearn.metrics import average_precision_score

import torch
import numpy as np
from sklearn.metrics import average_precision_score

def calculate_iou(preds, masks, smooth=1e-6):
    # Ensure inputs are binary
    preds = (preds > 0.5).float()
    masks = (masks > 0.5).float()

    intersection = (preds * masks).sum(dim=[1, 2, 3])  # Sum across H, W, C
    union = preds.sum(dim=[1, 2, 3]) + masks.sum(dim=[1, 2, 3]) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()  # Mean IoU over the batch

def calculate_dice(preds, masks, smooth=1e-6):
    # Assuming raw logits are passed here, apply sigmoid internally for Dice Coefficient
    preds = torch.sigmoid(preds)

    # Flatten label and prediction tensors
    preds = preds.view(-1)
    masks = masks.view(-1)

    intersection = (preds * masks).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + masks.sum() + smooth)
    return dice # This is the Dice Coefficient

def calculate_accuracy(preds, masks):
    # Ensure inputs are binary for accuracy calculation
    preds = (preds > 0.5).float()
    masks = (masks > 0.5).float()

    correct_pixels = (preds == masks).sum()
    total_pixels = torch.numel(preds) # Total number of elements in the tensor

    accuracy = correct_pixels.float() / total_pixels
    return accuracy

def calculate_ap(preds_logits, masks):
    # preds_logits: raw logits from the model output
    # masks: ground truth binary masks (0 or 1)

    # Apply sigmoid to get probabilities
    preds_prob = torch.sigmoid(preds_logits)

    # Flatten the tensors for AP calculation
    preds_flat = preds_prob.view(-1).cpu().numpy()
    masks_flat = masks.view(-1).cpu().numpy()

    # Calculate Average Precision
    # Handle cases where no positive samples are present
    if np.sum(masks_flat) == 0:
        return 0.0
    else:
        return average_precision_score(masks_flat, preds_flat)


file_path = "medical_segmentation_project/utils/metrics.py"


os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, "w") as f:
    f.write(metrics_content.strip())

print(f"Generated {file_path} with metric function definitions.")