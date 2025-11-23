"""UNet variant with only two downsampling stages but a deep bottleneck."""

import torch
import torch.nn as nn

from ..registry import register_model
from ..blocks.conv_blocks import DownSampling, Bottleneck, DoubleConv


@register_model("UNetTwoDown")
class UNetTwoDown(nn.Module):
    """
    UNet with two encoder downsampling steps while keeping a high-capacity bottleneck.

    Architecture overview (assuming 512x512 inputs):
        - Downsample x2: 512 -> 256 -> 128
        - Bottleneck processes features at 128x128 with deep channels
        - Upsample x2 back to 512 with skip connections
        - All convolutional blocks remain DoubleConv (baseline-style)
    """

    def __init__(self, config: dict):
        super().__init__()

        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        depths = config["depths"]  # e.g., [128, 256, 512] - bottleneck should be 512 to match baseline
        if len(depths) != 3:
            raise ValueError(
                "UNetTwoDown expects depths with three entries "
                "[encoder_level1, encoder_level2, bottleneck]"
            )
        dropout = config.get("dropout", 0.0)

        enc_ch1, enc_ch2, bott_ch = depths

        # Encoder (two downsampling stages)
        self.down_conv_1 = DownSampling(in_channels, enc_ch1, dropout=dropout)
        self.down_conv_2 = DownSampling(enc_ch1, enc_ch2, dropout=dropout)

        # Bottleneck keeps deep channel count (DoubleConv)
        self.bottleneck = Bottleneck(enc_ch2, bott_ch, dropout=dropout)

        # Decoder mirroring the encoder depth
        self.up_1 = nn.ConvTranspose2d(bott_ch, enc_ch2, kernel_size=2, stride=2)
        self.conv_1 = DoubleConv(enc_ch2 + enc_ch2, enc_ch2, dropout=dropout)

        self.up_2 = nn.ConvTranspose2d(enc_ch2, enc_ch1, kernel_size=2, stride=2)
        self.conv_2 = DoubleConv(enc_ch1 + enc_ch1, enc_ch1, dropout=dropout)

        # Output head
        self.out_conv = nn.Conv2d(enc_ch1, out_channels, kernel_size=1)

        if config.get("final_activation") == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1, x = self.down_conv_1(x)
        skip2, x = self.down_conv_2(x)

        x = self.bottleneck(x)

        x = self.up_1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv_1(x)

        x = self.up_2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv_2(x)

        x = self.out_conv(x)
        return self.final_activation(x)

