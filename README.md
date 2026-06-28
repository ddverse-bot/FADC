# FADC: Frequency-Adaptive and Anatomically-Calibrated Uncertainty Estimation for Medical Image Segmentation

Official PyTorch implementation of Frequency-Adaptive Dilated Convolution (FADC) and Contrastive Representation-based Anatomical Consistency (CRISP) for reliable medical image segmentation and uncertainty estimation.

##  Overview
This repository proposes a unified framework that combines:
* **Frequency-Adaptive Dilated Convolution (FADC)** for frequency-aware multi-scale feature extraction.
* **CRISP (Contrastive Representation-based Anatomical Consistency Estimation)** for anatomy-aware uncertainty estimation.
* **Soft-Binning Calibration Loss (ML1-ACE)** for confidence calibration.

The framework is evaluated on **BUSI, BUS-UC, and TN3K** datasets using multiple segmentation architectures: U-Net, U-Net++, Attention U-Net, and TransUNet.

##  Architecture
### Pipeline Overview
1. Input ultrasound image is passed through the segmentation network.
2. Predicted segmentation mask is generated.
3. Image and mask are encoded into a joint latent space.
4. Anatomical consistency is estimated using contrastive similarity.
5. Calibration and uncertainty estimation are performed.
6. Anatomically consistent uncertainty maps are generated.


