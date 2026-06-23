import torch
import torch.nn as nn

from .blocks import DoubleConv, Down, Up, OutConv


class AttentionGate(nn.Module):

    def __init__(self, Fg, Fl, Fint):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(Fg, Fint, 1),
            nn.BatchNorm2d(Fint)
        )

        self.Wx = nn.Sequential(
            nn.Conv2d(Fl, Fint, 1),
            nn.BatchNorm2d(Fint)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(Fint, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        psi = self.relu(
            self.Wg(g) + self.Wx(x)
        )

        psi = self.psi(psi)

        return x * psi
