# Medical Image Segmentation Project

This project focuses on medical image segmentation using U-Net architectures enhanced with advanced frequency-adaptive and attention-based modules. The goal is to accurately segment breast ultrasound images to assist in medical diagnosis.

## Project Structure

```
medical_segmentation_project/
├── checkpoints/                # Stores trained model weights
├── data/                       # Raw and processed datasets
│   └── README.md               # Instructions for data handling
├── models/                     # Custom modules and U-Net architectures
│   ├── custom_modules.py
│   └── unet_architectures.py
├── scripts/                    # Training and evaluation scripts
│   ├── train.py
│   └── evaluate.py
├── utils/                      # Utility functions (data prep, losses, metrics)
│   ├── data_preprocessing.py
│   ├── losses.py
│   └── metrics.py
├── main.py                     # Main entry point for project execution
└── requirements.txt            # Project dependencies
```

## 1. Project Setup

This project is organized into a modular structure to facilitate development, testing, and maintenance. The directory structure is outlined above.

## 2. Module Implementation

The `models/custom_modules.py` file contains the implementations of several custom PyTorch modules:

*   `Sobel`: Implements a Sobel filter for edge detection, used in frequency analysis.
*   `HybridFrequencyModule`: Combines Sobel filter output with a learnable convolutional layer to capture hybrid frequency features.
*   `FADC` (Frequency-Adaptive Dilated Convolution): Applies parallel dilated convolutions with adaptive weighting to capture multi-scale contextual information.
*   `Adakern`: Incorporates a Squeeze-and-Excitation (SE) block for adaptive channel-wise recalibration of feature responses.
*   `Freqselect`: Uses parallel convolutional branches with different kernel sizes (3, 5, 7) and an attention mechanism to adaptively weight the outputs.
*   `Adakern_DilatedLayer`: An AdaKern-style layer that supports dilation, combining adaptive channel weighting with multi-scale receptive fields.
*   `FADC_With_Adakern_Internal`: An enhanced FADC module where its internal dilated convolutions are replaced by `Adakern_DilatedLayer`s.

## 3. U-Net Architectures

The `models/unet_architectures.py` file defines three U-Net variants, integrating the custom modules:

*   `ComplexUNet`: A more U-Net where `FADC_With_Adakern_Internal` and `Freqselect` modules are used extensively in both encoder and decoder paths, as well as the bottleneck, to capture richer frequency-adaptive and attention-enhanced features.

## 4. Dataset Preparation

### Breast Ultrasound Images (BUSI) Dataset

The project uses the Breast Ultrasound Images (BUSI) dataset, available on Kaggle (`aryashah2k/breast-ultrasound-images-dataset`). This dataset contains ultrasound images of breast lesions (benign, malignant) and normal cases, along with their corresponding ground truth masks.

### Preprocessing (`utils/data_preprocessing.py`)

The `data_preprocessing.py` script handles:

*   Downloading the dataset from Kaggle (requires `kaggle.json` setup).
*   Splitting the dataset into training and validation sets.
*   Resizing images and masks to 256x256 pixels.
*   Converting masks to a binary format (0 or 1).
*   Creating PyTorch `Dataset` and `DataLoader` objects.

Refer to `data/README.md` for specific instructions on downloading the dataset.

## 5. Utility Functions

The `utils/` directory contains helper functions:

*   `losses.py`: Defines `DiceLoss` and `CombinedLoss` (a weighted sum of BCEWithLogitsLoss and DiceLoss) for model training.
*   `metrics.py`: Implements functions to calculate common segmentation metrics: IoU (Intersection over Union), Dice Coefficient, Accuracy, and Average Precision (AP).

## 6. Training Orchestration (`scripts/train.py`)

The `scripts/train.py` file provides a function (`train_model`) to train a specified model (typically `ComplexUNet`). It handles:

*   Moving the model and data to the appropriate device (CPU/GPU).
*   Iterating through epochs, performing forward and backward passes.
*   Calculating and logging training and validation loss/metrics.
*   Saving model checkpoints periodically to the `checkpoints/` directory.

## 7. Evaluation and Visualization (`scripts/evaluate.py`)

The `scripts/evaluate.py` file contains functions to:

*   Load a trained model from a checkpoint.
*   Evaluate the model on the validation set, computing all defined metrics (IoU, Dice, Accuracy, AP, FPS).
*   Visualize model predictions: original images, ground truth masks, predicted masks, internal feature maps (e.g., from the FADC bottleneck), and overlay visualizations for a selection of samples.

## 8. Dependencies

The `requirements.txt` file lists all necessary Python packages and their versions required to run this project. You can install them using:

```bash
pip install -r requirements.txt
```

### Usage:

To prepare the data:
```bash
python main.py --mode prepare_data


### Results:
Mean IoU: 0.6478

Mean Dice Score: 0.6851

Mean Accuracy: 0.9596

Mean Average Precision (AP): 0.7736

Mean FPS (on validation set): 1022.88

###Visualizations


![alt text](image.png)
![alt text](image-1.png)
![alt text](<Screenshot 2026-03-31 003051-1.png>)