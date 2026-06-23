import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Lightweight encoder used for both image and mask branches.
    """

    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.projection = nn.Linear(
            128,
            latent_dim
        )

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        z = self.projection(x)

        z = F.normalize(z, dim=1)

        return z


class CRISP(nn.Module):
    """
    Contrastive Representation-based
    Anatomical Consistency Estimation.

    Input:
        image
        predicted_mask

    Output:
        similarity score
        uncertainty score
    """

    def __init__(
        self,
        image_channels=1,
        mask_channels=1,
        latent_dim=128
    ):
        super().__init__()

        self.image_encoder = Encoder(
            image_channels,
            latent_dim
        )

        self.mask_encoder = Encoder(
            mask_channels,
            latent_dim
        )

    def forward(
        self,
        image,
        predicted_mask
    ):

        z_img = self.image_encoder(image)

        z_mask = self.mask_encoder(predicted_mask)

        similarity = F.cosine_similarity(
            z_img,
            z_mask,
            dim=1
        )

        uncertainty = 1.0 - similarity

        return {
            "image_embedding": z_img,
            "mask_embedding": z_mask,
            "similarity": similarity,
            "uncertainty": uncertainty
        }


class ContrastiveConsistencyLoss(nn.Module):
    """
    InfoNCE-style contrastive loss.

    Positive:
        image + correct mask

    Negative:
        image + incorrect mask
    """

    def __init__(
        self,
        temperature=0.07
    ):
        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        image_embedding,
        mask_embedding
    ):

        logits = (
            image_embedding
            @ mask_embedding.T
        ) / self.temperature

        labels = torch.arange(
            logits.size(0),
            device=logits.device
        )

        loss = F.cross_entropy(
            logits,
            labels
        )

        return loss


class AnatomicalConsistencyMap(nn.Module):
    """
    Produces a spatial consistency map.

    High values:
        Anatomically inconsistent regions

    Low values:
        Anatomically consistent regions
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        image,
        mask
    ):

        image_norm = (
            image - image.min()
        ) / (
            image.max() - image.min() + 1e-8
        )

        mask_norm = (
            mask - mask.min()
        ) / (
            mask.max() - mask.min() + 1e-8
        )

        consistency_map = torch.abs(
            image_norm - mask_norm
        )

        return consistency_map
