import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFADC(nn.Module):
    """
    Fourier-based Frequency Adaptive Dilated Convolution.

    Computes frequency content using FFT and dynamically selects
    a dilation rate based on the average frequency response.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        min_dilation=1,
        max_dilation=4
    ):
        super().__init__()

        self.min_dilation = min_dilation
        self.max_dilation = max_dilation

        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size
            )
        )

        self.bias = nn.Parameter(
            torch.zeros(out_channels)
        )

    def compute_frequency_map(self, x):

        x_gray = torch.mean(x, dim=1, keepdim=True)

        fft = torch.fft.fft2(x_gray)

        fft_shift = torch.fft.fftshift(fft)

        magnitude = torch.abs(fft_shift)

        frequency_map = torch.log1p(magnitude)

        return frequency_map

    def compute_adaptive_dilation(self, frequency_map):

        freq_norm = (
            frequency_map - frequency_map.min()
        ) / (
            frequency_map.max()
            - frequency_map.min()
            + 1e-6
        )

        dilation_map = (
            self.max_dilation
            - freq_norm
            * (self.max_dilation - self.min_dilation)
        )

        adaptive_dilation = int(
            torch.round(
                dilation_map.mean()
            ).item()
        )

        adaptive_dilation = max(
            self.min_dilation,
            min(self.max_dilation, adaptive_dilation)
        )

        return adaptive_dilation

    def forward(self, x):

        frequency_map = self.compute_frequency_map(x)

        adaptive_dilation = self.compute_adaptive_dilation(
            frequency_map
        )

        padding = adaptive_dilation

        out = F.conv2d(
            x,
            self.weight,
            self.bias,
            padding=padding,
            dilation=adaptive_dilation
        )

        return out
