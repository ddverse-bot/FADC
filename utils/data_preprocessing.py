import os
import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


import os
import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "/content/Dataset_BUSI_with_GT" 

class BUSI(Dataset):
    def __init__(self, img_dir, msk_dir, transform=None):
        self.names = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.transform = transform

    def __len__(self): return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img_path = os.path.join(self.img_dir, n)
        msk_path = os.path.join(self.msk_dir, n)

        img = cv2.resize(cv2.imread(img_path), (256, 256))
        msk = cv2.resize(cv2.imread(msk_path, 0), (256, 256), interpolation=cv2.INTER_NEAREST)

        # Convert mask to binary (0 or 1)
        msk = (msk > 127).astype('float32')

        # Convert to PyTorch tensors and normalize image
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.
        msk = torch.tensor(msk).unsqueeze(0) # Add channel dimension for mask

        if self.transform:
            img, msk = self.transform(img, msk)

        return img, msk

def save_split(imgs, msks, split_name):
    # Ensure output directories exist
    os.makedirs(f"/content/BUSI/images/{split_name}", exist_ok=True)
    os.makedirs(f"/content/BUSI/masks/{split_name}", exist_ok=True)

    print(f"Saving {split_name} split to /content/BUSI/images/{split_name} and /content/BUSI/masks/{split_name}")
    for k, (img_p, msk_p) in enumerate(zip(imgs, msks)):
        img = cv2.resize(cv2.imread(img_p), (256, 256))
        msk = cv2.resize(cv2.imread(msk_p, 0), (256, 256), interpolation=cv2.INTER_NEAREST)
        msk = (msk > 127).astype(np.uint8) * 255 # Ensure mask is binary 0 or 255 for saving

        cv2.imwrite(f"/content/BUSI/images/{split_name}/{k}.png", img)
        cv2.imwrite(f"/content/BUSI/masks/{split_name}/{k}.png", msk)

def prepare_busi_dataset(data_dir=DATA_DIR, test_size=0.2, random_state=42):
    images, masks = [], []
    for cat in ["benign", "malignant", "normal"]:
        folder = os.path.join(data_dir, cat)
        for f in os.listdir(folder):
            if "_mask" not in f: # Only consider original image files
                img_p = os.path.join(folder, f)
                m_p = os.path.join(folder, f.replace(".png", "_mask.png"))
                if os.path.exists(m_p):
                    images.append(img_p)
                    masks.append(m_p)

    print(f"Found {len(images)} image-mask pairs.")

    # Split data into training and validation sets
    tr_i, va_i, tr_m, va_m = train_test_split(images, masks, test_size=test_size, random_state=random_state)

    print(f"Training images: {len(tr_i)}, Validation images: {len(va_i)}")


    save_split(tr_i, tr_m, "train")
    save_split(va_i, va_m, "val")

    # Create DataLoaders
    train_loader = DataLoader(BUSI("/content/BUSI/images/train", "/content/BUSI/masks/train"),
                              batch_size=4, shuffle=True)
    val_loader   = DataLoader(BUSI("/content/BUSI/images/val", "/content/BUSI/masks/val"),
                              batch_size=4, shuffle=False) # No need to shuffle validation data
    
    return train_loader, val_loader


if __name__ == '__main__':
    print("Starting dataset preparation...")
    # This part assumes the dataset is already downloaded and unzipped at DATA_DIR
    # It needs to be run once to generate the /content/BUSI/ structure
    # train_loader, val_loader = prepare_busi_dataset()
    # print("Dataset prepared and DataLoaders created.")

os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, "w") as f:
    f.write(data_preprocessing_content.strip())


print(f"Generated {file_path} with data preprocessing utilities.")