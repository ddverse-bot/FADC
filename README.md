# FADC: Frequency-Adaptive and Anatomically-Calibrated Uncertainty Estimation for Medical Image Segmentation

Official PyTorch implementation of **Frequency-Adaptive Dilated Convolution (FADC)**, **Contrastive Representation-based Anatomical Consistency (CRISP)**, and **Soft-Binning Calibration Loss (ML1-ACE)** for reliable medical image segmentation and uncertainty estimation.

---

# Overview

Deep learning-based medical image segmentation methods have demonstrated remarkable performance across various medical imaging tasks. However, these models often suffer from two major limitations:

1. Poor calibration and overconfident predictions.
2. Unreliable uncertainty estimation, particularly in challenging imaging scenarios such as ultrasound.

This repository proposes a unified framework that combines:

- Frequency-Adaptive Dilated Convolution (FADC)
- Contrastive Representation-based Anatomical Consistency Estimation (CRISP)
- Soft-Binning Calibration Loss (ML1-ACE)

to improve segmentation accuracy, uncertainty estimation, and confidence calibration.

The proposed framework is evaluated on:

- BUSI
- BUS-UC
- TN3K

using multiple segmentation backbones and their corresponding FADCs with loss computations:

- U-Net
- U-Net++
- Attention U-Net
- TransUNet



# Frequency-Adaptive Dilated Convolution (FADC)

---

## Method

Given an intermediate feature map

```math
X \in \mathbb{R}^{C\times H\times W}
```

we compute its Fourier representation:

```math
F = \mathcal{F}(X)
```

and obtain the magnitude spectrum:

```math
M = \log(1 + |F|)
```

The magnitude spectrum is normalized:

```math
\hat{M}
=
\frac{M-M_{min}}
     {M_{max}-M_{min}}
```

The adaptive dilation rate is then computed as:

```math
d
=
d_{max}
-
\hat{M}
(d_{max}-d_{min})
```

where:

- high-frequency regions receive smaller dilation rates,
- low-frequency regions receive larger dilation rates.

The final convolution becomes:

```math
Y
=
Conv(X;W,d)
```

---

## Advantages of FADC

- Dynamic receptive field adaptation.
- Improved multi-scale feature extraction.
- Better boundary delineation.
- Improved segmentation of irregular lesions.
- More reliable uncertainty estimation.

---

# Contrastive Representation-based Anatomical Consistency (CRISP)

CRISP introduces anatomy-aware uncertainty estimation.

Given an image and its predicted segmentation mask:

```math
z_I = E_I(I)
```

```math
z_M = E_M(M)
```

The anatomical consistency score is computed using cosine similarity:

```math
s
=
\frac{
z_I \cdot z_M
}{
||z_I||
||z_M||
}
```

The uncertainty score is then defined as:

```math
u = 1-s
```

Regions exhibiting poor anatomical agreement receive higher uncertainty values.

---

# Soft-Binning Calibration Loss (ML1-ACE)

Deep segmentation models are frequently poorly calibrated.

Given predicted probabilities:

```math
p = \sigma(z)
```

soft assignment weights are computed:

```math
w_k
=
\exp
\left(
-\frac{
(p-c_k)^2
}{
2\sigma^2
}
\right)
```

where

```math
c_k
```

denotes the center of the k-th calibration bin.

The calibration loss is:

```math
L_{ACE}
=
\sum_k
\bar{w}_k
|
Acc_k-Conf_k
|
```

which encourages confidence estimates to align with prediction accuracy.

---

# Overall Framework

<p align="center">
<img width="1458" height="720" alt="abstractdiaa" src="https://github.com/user-attachments/assets/d6eb68f2-3863-4a9c-9fcf-03d5508c903d" />
</p>

The proposed framework combines:

- Frequency-aware feature extraction.
- Anatomy-aware uncertainty estimation.
- Confidence calibration.

to produce accurate and reliable medical image segmentation and uncertainty estimates.

---

The proposed framework progressively improves:

- Segmentation quality
- Boundary delineation
- Anatomical consistency
- Confidence calibration
- Uncertainty estimation

---



# Repository Structure

```text
medical_segmentation_project/
├── checkpoints/
├── data/
├── models/
│   ├── blocks.py
│   ├── fourier_fadc.py
│   ├── crisp.py
│   ├── calibration.py
│   ├── unet.py
│   ├── unetpp.py
│   ├── attention_unet.py
│   ├── transunet.py
│   └── __init__.py
│
├── scripts/
│   ├── train.py
│   └── evaluate.py
│
├── utils/
│   ├── data_preprocessing.py
│   ├── losses.py
│   ├── metrics.py
│   └── visualization.py
│
├── figures/
│   ├── architecture.png
│   ├── fadc_module.png
│   ├── crisp_pipeline.png
│   ├── qualitative_results.png
│   └── results_table.png
│
├── requirements.txt
├── main.py
└── README.md
```

---

# Installation

```bash
git clone https://github.com/ddverse-bot/FADC.git
cd FADC

conda create -n fadc python=3.10
conda activate fadc

pip install -r requirements.txt
```

---


# Training

```bash
python scripts/train.py --model unet
python scripts/train.py --model unet --fadc
python scripts/train.py --model unet --fadc --calibration

python scripts/train.py --model unetpp
python scripts/train.py --model attunet
python scripts/train.py --model transunet
```

---

# Evaluation

```bash
python scripts/evaluate.py \
--model unetpp \
--checkpoint checkpoints/unetpp_fadc_calibration.pth
```

---

# Evaluation Metrics

## Segmentation

- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Accuracy
- Average Precision (AP)

## Calibration

- Expected Calibration Error (ECE)
- Adaptive Calibration Error (ACE)
- Brier Score (BS)

## Uncertainty

- Entropy
- Error-Uncertainty Correlation

---

# Acknowledgements

This repository builds upon:

- PyTorch
- segmentation_models_pytorch
- U-Net
- U-Net++
- Attention U-Net
- TransUNet
- Metrics Reloaded
- CRISP
- Medical image uncertainty estimation literature.
