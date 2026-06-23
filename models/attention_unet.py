import torch
import torch.nn as nn

from .blocks import DoubleConv, Down, Up, OutConv


class AttentionGate(nn.Module):

    def __init__(self, Fg, Fl, Fint):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(Fg, Fint, kernel_size=1),
            nn.BatchNorm2d(Fint)
        )

        self.Wx = nn.Sequential(
            nn.Conv2d(Fl, Fint, kernel_size=1),
            nn.BatchNorm2d(Fint)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(Fint, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.Wg(g)
        x1 = self.Wx(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


#########################################################
# BASELINE ATTENTION U-NET
#########################################################

class AttentionUNet(nn.Module):

    def __init__(
        self,
        n_channels=1,
        n_classes=1
    ):
        super().__init__()

        self.inc = DoubleConv(
            n_channels,
            64
        )

        self.down1 = Down(
            64,
            128,
            use_fadc=False
        )

        self.down2 = Down(
            128,
            256,
            use_fadc=False
        )

        self.down3 = Down(
            256,
            512,
            use_fadc=False
        )

        self.down4 = Down(
            512,
            1024,
            use_fadc=False
        )

        self.att4 = AttentionGate(
            1024,
            512,
            256
        )

        self.att3 = AttentionGate(
            512,
            256,
            128
        )

        self.att2 = AttentionGate(
            256,
            128,
            64
        )

        self.att1 = AttentionGate(
            128,
            64,
            32
        )

        self.up1 = Up(
            1024,
            512
        )

        self.up2 = Up(
            512,
            256
        )

        self.up3 = Up(
            256,
            128
        )

        self.up4 = Up(
            128,
            64
        )

        self.outc = OutConv(
            64,
            n_classes
        )

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x4 = self.att4(x5, x4)
        x = self.up1(x5, x4)

        x3 = self.att3(x, x3)
        x = self.up2(x, x3)

        x2 = self.att2(x, x2)
        x = self.up3(x, x2)

        x1 = self.att1(x, x1)
        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits


#########################################################
# ATTENTION U-NET + FADC
#########################################################

class AttentionUNetFADC(nn.Module):

    def __init__(
        self,
        n_channels=1,
        n_classes=1
    ):
        super().__init__()

        self.inc = DoubleConv(
            n_channels,
            64
        )

        # shallow encoder remains standard

        self.down1 = Down(
            64,
            128,
            use_fadc=False
        )

        # deep encoder uses FADC

        self.down2 = Down(
            128,
            256,
            use_fadc=True
        )

        self.down3 = Down(
            256,
            512,
            use_fadc=True
        )

        self.down4 = Down(
            512,
            1024,
            use_fadc=True
        )

        self.att4 = AttentionGate(
            1024,
            512,
            256
        )

        self.att3 = AttentionGate(
            512,
            256,
            128
        )

        self.att2 = AttentionGate(
            256,
            128,
            64
        )

        self.att1 = AttentionGate(
            128,
            64,
            32
        )

        self.up1 = Up(
            1024,
            512
        )

        self.up2 = Up(
            512,
            256
        )

        self.up3 = Up(
            256,
            128
        )

        self.up4 = Up(
            128,
            64
        )

        self.outc = OutConv(
            64,
            n_classes
        )

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x4 = self.att4(x5, x4)
        x = self.up1(x5, x4)

        x3 = self.att3(x, x3)
        x = self.up2(x, x3)

        x2 = self.att2(x, x2)
        x = self.up3(x, x2)

        x1 = self.att1(x, x1)
        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits
