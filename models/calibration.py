import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftBinningCalibrationLoss(nn.Module):

    def __init__(
        self,
        num_bins=15,
        sigma=0.05
    ):
        super().__init__()

        self.num_bins = num_bins
        self.sigma = sigma

        centers = torch.linspace(
            0.0,
            1.0,
            num_bins
        )

        self.register_buffer(
            "bin_centers",
            centers
        )

    def forward(
        self,
        logits,
        targets
    ):

        probs = torch.sigmoid(logits)

        probs = probs.reshape(-1)
        targets = targets.reshape(-1).float()

        ece = 0.0

        for center in self.bin_centers:

            weights = torch.exp(
                -((probs - center) ** 2)
                /
                (2 * self.sigma ** 2)
            )

            weights = weights + 1e-8

            conf = (
                weights * probs
            ).sum() / weights.sum()

            acc = (
                weights * targets
            ).sum() / weights.sum()

            ece += (
                weights.mean()
                *
                torch.abs(acc - conf)
            )

        return ece
