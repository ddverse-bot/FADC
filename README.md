# FADC: Frequency-Adaptive and Anatomically-Calibrated Uncertainty Estimation for Medical Image Segmentation

Official PyTorch implementation of Frequency-Adaptive Dilated Convolution (FADC) and Contrastive Representation-based Anatomical Consistency (CRISP) for reliable medical image segmentation and uncertainty estimation.

##  Overview
This repository proposes a unified framework that combines:
* **Frequency-Adaptive Dilated Convolution (FADC)** for frequency-aware multi-scale feature extraction.
* **CRISP (Contrastive Representation-based Anatomical Consistency Estimation)** for anatomy-aware uncertainty estimation.
* **Soft-Binning Calibration Loss (ML1-ACE)** for confidence calibration.

The framework is evaluated on **BUSI, BUS-UC, and TN3K** datasets using multiple segmentation architectures: U-Net, U-Net++, Attention U-Net, and TransUNet.

##  Architecture

git clone [https://github.com/ddverse-bot/FADC.git](https://github.com/ddverse-bot/FADC.git)
cd FADC
pip install -r requirements.txt
# Train with FADC and Calibration
python scripts/train.py --model unetpp --fadc --calibration

# Train individual architectures
python scripts/train.py --model unet
python scripts/train.py --model unetpp
python scripts/train.py --model attunet
python scripts/train.py --model transunet
python scripts/evaluate.py --model unetpp --checkpoint checkpoints/unetpp_fadc_calibration.pth

