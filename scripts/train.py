import os


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

# Import necessary modules from the project structure
from medical_segmentation_project.models.unet_architectures import ComplexUNet
from medical_segmentation_project.utils.data_preprocessing import prepare_busi_dataset
from medical_segmentation_project.utils.losses import DiceLoss, CombinedLoss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cuda'):
    model.to(device)

    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    model_save_dir = os.path.join("medical_segmentation_project", "checkpoints")
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, "complex_unet_segmentation_model.pth")

    print(f"Training started on device: {device}")
    print(f"Model checkpoints will be saved to: {model_save_path}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train")
        for inputs, masks in train_loop:
            inputs, masks = inputs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_loop.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice_score = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Val  ")
        with torch.no_grad():
            for inputs, masks in val_loop:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * inputs.size(0)

                preds = torch.sigmoid(outputs)
                dice = 1 - DiceLoss()(preds, masks) # DiceLoss instantiated here
                val_dice_score += dice.item() * inputs.size(0)
                val_loop.set_postfix(loss=loss.item(), dice=dice.item())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_dice = val_dice_score / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        history['val_dice'].append(epoch_val_dice)

        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Dice: {epoch_val_dice:.4f}")

        # Save model checkpoint after each epoch
        torch.save(model.state_dict(), model_save_path)
        print(f"Model checkpoint saved to {model_save_path}")

    print("Training complete!")
    return model, history


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    print("Preparing dataset...")
    
    try:
        train_loader, val_loader = prepare_busi_dataset()
        print("Dataset prepared and DataLoaders created.")
    except Exception as e:
        print(f"Error preparing dataset. Make sure `kaggle.json` is configured and dataset is downloaded and unzipped. Error: {e}")
        exit() 

    # Instantiate the ComplexUNet model
    model = ComplexUNet(in_channels=3, out_channels=1)
    model.to(device)

    # Define loss function and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Path for model weights
    model_weights_dir = os.path.join("medical_segmentation_project", "checkpoints")
    os.makedirs(model_weights_dir, exist_ok=True) # Ensure directory exists
    model_weights_path = os.path.join(model_weights_dir, "complex_unet_segmentation_model.pth")

    # Load existing model weights if available to resume training
    if os.path.exists(model_weights_path):
        print(f"Loading previous weights from {model_weights_path} to resume training.")
        model.load_state_dict(torch.load(model_weights_path, map_location=device), strict=False)
    else:
        print(f"No previous weights found at {model_weights_path}. Training from scratch.")

    # Train the model
    print("\n--- Starting ComplexUNet training ---")
    trained_model, training_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device=device
    )

    print("\nTraining History:")
    for k, v in training_history.items():
        print(f"{k}: {v}")

    
    torch.save(trained_model.state_dict(), model_weights_path)
    print(f"Final trained model saved as {model_weights_path}")



file_path = "medical_segmentation_project/scripts/train.py"

os.makedirs(os.path.dirname(file_path), exist_ok=True)


with open(file_path, "w") as f:
    f.write(train_script_content.strip())

print(f"Generated {file_path} with the training script.")