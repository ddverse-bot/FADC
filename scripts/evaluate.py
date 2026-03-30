import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import random

# Import necessary modules from the project structure
from medical_segmentation_project.models.unet_architectures import ComplexUNet
from medical_segmentation_project.utils.data_preprocessing import BUSI # Only BUSI Dataset needed for evaluation
from medical_segmentation_project.utils.metrics import calculate_iou, calculate_dice, calculate_accuracy, calculate_ap


val_loader_dir = "/content/BUSI/images/val"
val_masks_dir = "/content/BUSI/masks/val"

if not os.path.exists(val_loader_dir) or not os.path.exists(val_masks_dir):
    print("Warning: Validation data directories not found. Please run `train.py` or `data_preprocessing.py` first to prepare the dataset.")
    

val_loader = torch.utils.data.DataLoader(BUSI(val_loader_dir, val_masks_dir),
                                      batch_size=4, shuffle=False)

def evaluate_model(model, val_loader, device='cuda'):
    model.to(device)
    model.eval() # Set model to evaluation mode

    all_iou = []
    all_dice = []
    all_accuracy = []
    all_ap = []
    all_fps = []

    print("\n--- Starting Model Evaluation ---")
    with torch.no_grad():
        for inputs, masks in val_loader:
            inputs, masks = inputs.to(device), masks.to(device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            batch_size = inputs.size(0)
            fps = batch_size / (end_time - start_time)
            all_fps.append(fps)

            preds_prob = torch.sigmoid(outputs)
            preds_binary = (preds_prob > 0.5).float()

            iou = calculate_iou(preds_binary, masks)
            dice = calculate_dice(outputs, masks)
            accuracy = calculate_accuracy(preds_binary, masks)
            ap = calculate_ap(outputs, masks)

            all_iou.append(iou.item())
            all_dice.append(dice.item())
            all_accuracy.append(accuracy.item())
            all_ap.append(ap)

    mean_iou = np.mean(all_iou)
    mean_dice = np.mean(all_dice)
    mean_accuracy = np.mean(all_accuracy)
    mean_ap = np.mean(all_ap)
    mean_fps = np.mean(all_fps)

    metrics = {
        "Mean IoU": mean_iou,
        "Mean Dice Score": mean_dice,
        "Mean Accuracy": mean_accuracy,
        "Mean Average Precision (AP)": mean_ap,
        "Mean FPS": mean_fps
    }

    print(f"\n--- Evaluation Results ---")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    return metrics

def visualize_predictions(model, val_loader, device='cuda', num_samples=5, smoothing_sigma=1.5):
    model.to(device)
    model.eval()

    print(f"\n--- Generating Visualizations for {num_samples} Random Samples ---")
    val_dataset_indices = list(range(len(val_loader.dataset)))
    random_indices = random.sample(val_dataset_indices, min(num_samples, len(val_dataset_indices)))

    plt.figure(figsize=(20, 4 * num_samples))

    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            sample_image, sample_mask = val_loader.dataset[idx]
            sample_image = sample_image.unsqueeze(0).to(device)
            sample_mask  = sample_mask.unsqueeze(0).to(device)

            # Manually trace forward pass to extract FADC's intermediate outputs for visualization
            x = sample_image
            skip_connections = []

            # Encoder path
            for down_block_seq in model.downs:
                x = down_block_seq(x)
                skip_connections.append(x)
                x = model.pool(x)

            
            bottleneck_output = model.bottleneck(x)
           
            fadc_map_np = bottleneck_output.mean(dim=1).squeeze().cpu().numpy() 

            # Decoder path
            x = bottleneck_output
            skip_connections = skip_connections[::-1]

            for j in range(0, len(model.ups), 2):
                x = model.ups[j](x)
                skip_idx = (j // 2)
                skip_connection = skip_connections[skip_idx]

                if x.shape[2:] != skip_connection.shape[2:]:
                    x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = model.ups[j+1](concat_skip)

            prediction_logits = model.final_conv(x)
            prediction = torch.sigmoid(prediction_logits)
            prediction_binary = (prediction > 0.5).float()

            original_image_np = sample_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            ground_truth_np   = sample_mask.squeeze(0).squeeze(0).cpu().numpy()

            prediction_np_smoothed = cv2.GaussianBlur(prediction_binary.squeeze(0).squeeze(0).cpu().numpy(), (0,0), smoothing_sigma)
            prediction_np = (prediction_np_smoothed > 0.5).astype(np.float32)

            overlay_image_np = original_image_np.copy()
            overlay_mask_area = prediction_np > 0
            if overlay_image_np.ndim == 2:
                overlay_image_np = np.stack([overlay_image_np, overlay_image_np, overlay_image_np], axis=-1)
            overlay_image_np[overlay_mask_area, 1] = 1.0
            overlay_image_np[overlay_mask_area, 0] *= 0.5
            overlay_image_np[overlay_mask_area, 2] *= 0.5

            row_start = i * 5

            plt.subplot(num_samples, 5, row_start + 1)
            plt.imshow(original_image_np)
            plt.title(f'Sample {idx}\nOriginal Image')
            plt.axis('off')

            plt.subplot(num_samples, 5, row_start + 2)
            plt.imshow(ground_truth_np, cmap='gray')
            plt.title(f'Sample {idx}\nGround Truth')
            plt.axis('off')

            plt.subplot(num_samples, 5, row_start + 3)
            plt.imshow(prediction_np, cmap='gray')
            plt.title(f'Sample {idx}\nPredicted Mask')
            plt.axis('off')

            plt.subplot(num_samples, 5, row_start + 4)
            plt.imshow(fadc_map_np, cmap='viridis')
            plt.title(f'Sample {idx}\nBottleneck Map')
            plt.axis('off')

            plt.subplot(num_samples, 5, row_start + 5)
            plt.imshow(overlay_image_np)
            plt.title(f'Sample {idx}\nOverlay')
            plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the ComplexUNet model
    model = ComplexUNet(in_channels=3, out_channels=1)
    model.to(device)

    # Path to load model weights
    model_weights_path = os.path.join("medical_segmentation_project", "checkpoints", "complex_unet_segmentation_model.pth")

    if os.path.exists(model_weights_path):
        print(f"Loading model weights from {model_weights_path}")
        model.load_state_dict(torch.load(model_weights_path, map_location=device), strict=False)
    else:
        print(f"Error: Model weights not found at {model_weights_path}. Please train the model first.")
        exit() # Exit if model weights are not found

  
    evaluation_metrics = evaluate_model(model, val_loader, device=device)

    visualize_predictions(model, val_loader, device=device, num_samples=10)


file_path = "medical_segmentation_project/scripts/evaluate.py"

os.makedirs(os.path.dirname(file_path), exist_ok=True)


with open(file_path, "w") as f:
    f.write(evaluate_script_content.strip())

print(f"Generated {file_path} with the evaluation script.")
