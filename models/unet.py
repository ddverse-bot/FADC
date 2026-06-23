import torch.nn as nn

from .blocks import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):

    def __init__(
        self,
        n_channels=1,
        n_classes=1
    ):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64)

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

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        return self.outc(x)


class UNetFADC(nn.Module):

    def __init__(
        self,
        n_channels=1,
        n_classes=1
    ):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64)

        # shallow encoder stays standard

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

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        return self.outc(x)
