
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom modules
from medical_segmentation_project.models.custom_modules import (
    FADC, Adakern, Freqselect, Adakern_DilatedLayer, FADC_With_Adakern_Internal
)

# --- Basic Building Block --- #

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# --- U-Net Architectures --- #

class SimpleUNetFADC(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(SimpleUNetFADC, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling path)
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # Bottleneck with FADC
        self.bottleneck_fadc = FADC(in_channels=features[-1], out_channels=features[-1], kernel_size=3, dilation_rates=[1, 2, 4])

        # Decoder (Upsampling path)
        for i in range(len(features) - 1, -1, -1):
            if i == len(features) - 1: # First upsampling step (from bottleneck)
                ups_input_channels = features[i]
                output_channels_for_conv_transpose = features[i-1]
                skip_channels = features[i]
            elif i > 0: # Intermediate upsampling steps
                ups_input_channels = features[i]
                output_channels_for_conv_transpose = features[i-1]
                skip_channels = features[i]
            else: # Last upsampling step (i = 0)
                ups_input_channels = features[i]
                output_channels_for_conv_transpose = features[i]
                skip_channels = features[i]

            # ConvTranspose2d
            self.ups.append(
                nn.ConvTranspose2d(ups_input_channels, output_channels_for_conv_transpose, kernel_size=2, stride=2)
            )

            # ConvBlock (after concatenation)
            conv_block_in_channels = output_channels_for_conv_transpose + skip_channels
            conv_block_out_channels = output_channels_for_conv_transpose
            self.ups.append(ConvBlock(conv_block_in_channels, conv_block_out_channels))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck_fadc(x)

        skip_connections = skip_connections[::-1] # Reverse for upsampling path

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) # ConvTranspose2d

            skip_idx = (i // 2)
            skip_connection = skip_connections[skip_idx]

            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip) # ConvBlock

        return self.final_conv(x)


class SimpleUNetAdakern(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(SimpleUNetAdakern, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling path)
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # Bottleneck with Adakern
        self.bottleneck_adakern = Adakern(in_channels=features[-1], out_channels=features[-1])

        # Decoder (Upsampling path)
        for i in range(len(features) - 1, -1, -1):
            if i == len(features) - 1: # First upsampling step (from bottleneck)
                ups_input_channels = features[i]
                output_channels_for_conv_transpose = features[i-1]
                skip_channels = features[i]
            elif i > 0: # Intermediate upsampling steps
                ups_input_channels = features[i]
                output_channels_for_conv_transpose = features[i-1]
                skip_channels = features[i]
            else: # Last upsampling step (i = 0)
                ups_input_channels = features[i]
                output_channels_for_conv_transpose = features[i]
                skip_channels = features[i]

            # ConvTranspose2d
            self.ups.append(
                nn.ConvTranspose2d(ups_input_channels, output_channels_for_conv_transpose, kernel_size=2, stride=2)
            )

            # ConvBlock (after concatenation)
            conv_block_in_channels = output_channels_for_conv_transpose + skip_channels
            conv_block_out_channels = output_channels_for_conv_transpose
            self.ups.append(ConvBlock(conv_block_in_channels, conv_block_out_channels))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck_adakern(x)

        skip_connections = skip_connections[::-1] # Reverse for upsampling path

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)

            skip_idx = (i // 2)
            skip_connection = skip_connections[skip_idx]

            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip)

        return self.final_conv(x)


class SimpleUNetFreqselect(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(SimpleUNetFreqselect, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling path)
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # Bottleneck with Freqselect
        self.bottleneck_freqselect = Freqselect(in_channels=features[-1], out_channels=features[-1])

        # Decoder (Upsampling path)
        for i in range(len(features) - 1, -1, -1):
            if i == len(features) - 1: # First upsampling step (from bottleneck)
                ups_input_channels = features[i]
                output_channels_for_conv_transpose = features[i-1]
                skip_channels = features[i]
            elif i > 0: # Intermediate upsampling steps
                ups_input_channels = features[i]
                output_channels_for_conv_transpose = features[i-1]
                skip_channels = features[i]
            else: # Last upsampling step (i = 0)
                ups_input_channels = features[i]
                output_channels_for_conv_transpose = features[i]
                skip_channels = features[i]

            # ConvTranspose2d
            self.ups.append(
                nn.ConvTranspose2d(ups_input_channels, output_channels_for_conv_transpose, kernel_size=2, stride=2)
            )

            # ConvBlock (after concatenation)
            conv_block_in_channels = output_channels_for_conv_transpose + skip_channels
            conv_block_out_channels = output_channels_for_conv_transpose
            self.ups.append(ConvBlock(conv_block_in_channels, conv_block_out_channels))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck_freqselect(x)

        skip_connections = skip_connections[::-1] # Reverse for upsampling path

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)

            skip_idx = (i // 2)
            skip_connection = skip_connections[skip_idx]

            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip)

        return self.final_conv(x)


class ComplexUNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(ComplexUNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling path): Each block is FADC_With_Adakern_Internal + Freqselect
        for feature in features:
            self.downs.append(nn.Sequential(
                FADC_With_Adakern_Internal(in_channels, feature),
                Freqselect(feature, feature)
            ))
            in_channels = feature

        # Bottleneck: FADC_With_Adakern_Internal + Freqselect
        self.bottleneck = nn.Sequential(
            FADC_With_Adakern_Internal(features[-1], features[-1]),
            Freqselect(features[-1], features[-1])
        )

        # Decoder (Upsampling path): Each block is ConvTranspose2d + FADC_With_Adakern_Internal + Freqselect
        for i in range(0, len(features)):
            # Determine input/output channels for ConvTranspose2d and skip connection
            ups_input_channels = features[len(features) - 1 - i] if i < len(features) else features[0]
            output_channels_for_conv_transpose = features[len(features) - 2 - i] if (len(features) - 2 - i) >= 0 else features[0]
            skip_channels = features[len(features) - 1 - i] if i > 0 else features[0]

            self.ups.append(
                nn.ConvTranspose2d(ups_input_channels, output_channels_for_conv_transpose, kernel_size=2, stride=2)
            )
            # The next sequential block takes (upsampled + skip) channels and outputs 'output_channels_for_conv_transpose'
            conv_block_in_channels = output_channels_for_conv_transpose + skip_channels
            conv_block_out_channels = output_channels_for_conv_transpose
            self.ups.append(nn.Sequential(
                FADC_With_Adakern_Internal(conv_block_in_channels, conv_block_out_channels),
                Freqselect(conv_block_out_channels, conv_block_out_channels)
            ))


        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        # Encoder
        for down_block_seq in self.downs:
            x = down_block_seq(x) # Output is from Freqselect
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x) # Output is from Freqselect

        skip_connections = skip_connections[::-1] # Reverse for upsampling path

        # Decoder
        # The loop runs for len(features) * 2 iterations, two modules per step in self.ups
        # So it handles the ConvTranspose2d and the subsequent Sequential block
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) # ConvTranspose2d

            skip_idx = (i // 2)
            if skip_idx < len(skip_connections):
                skip_connection = skip_connections[skip_idx]
            else:
                # This handles cases where the features list might be shorter than expected for decoder
                # or if the indexing logic is slightly off at the very end.
                # For a typical U-Net this should not be hit.
                print(f"Warning: skip_idx {skip_idx} out of bounds for skip_connections list of length {len(skip_connections)}")
                break # Exit loop if skip connection isn't available

            # Handle potential spatial dimension mismatch after ConvTranspose2d
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip) # FADC_With_Adakern_Internal + Freqselect

        return self.final_conv(x)
