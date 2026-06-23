import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from .fourier_fadc import FourierFADC

class TransUNet(nn.Module):

```
def __init__(
    self,
    n_channels=1,
    n_classes=1
):
    super().__init__()

    self.model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=n_channels,
        classes=n_classes
    )

def forward(self, x):

    return self.model(x)
```

class TransUNetFADC(nn.Module):

```
def __init__(
    self,
    n_channels=1,
    n_classes=1
):
    super().__init__()

    self.encoder = smp.encoders.get_encoder(
        "resnet50",
        in_channels=n_channels,
        depth=5,
        weights="imagenet"
    )

    self.fadc3 = FourierFADC(
        1024,
        1024
    )

    self.fadc4 = FourierFADC(
        2048,
        2048
    )

    self.decoder = smp.decoders.unet.decoder.UnetDecoder(
        encoder_channels=self.encoder.out_channels,
        decoder_channels=(256, 128, 64, 32, 16),
        n_blocks=5,
        use_batchnorm=True,
        center=False,
        attention_type=None,
    )

    self.segmentation_head = nn.Conv2d(
        16,
        n_classes,
        kernel_size=1
    )

def forward(self, x):

    features = self.encoder(x)

    features = list(features)

    features[-2] = self.fadc3(features[-2])

    features[-1] = self.fadc4(features[-1])

    decoder_output = self.decoder(*features)

    masks = self.segmentation_head(
        decoder_output
    )

    return masks
```
