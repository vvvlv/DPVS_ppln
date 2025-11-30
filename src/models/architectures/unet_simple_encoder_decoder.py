"""UNet architecture with simplified encoder and decoder (SingleConv blocks)."""

import torch
import torch.nn as nn

from ..registry import register_model
from ..blocks.conv_blocks import SingleConv, Bottleneck


class SimpleDownSampling(nn.Module):
    """Downsampling block using SingleConv followed by max pooling."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        super().__init__()
        self.single_conv = SingleConv(input_size, output_size, dropout=dropout)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        conv_x = self.single_conv(x)
        x = self.max_pool(conv_x)
        return conv_x, x


class SimpleUpSampling(nn.Module):
    """Upsampling block using transpose conv followed by SingleConv."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=2,
            stride=2,
        )
        self.single_conv = SingleConv(input_size, output_size, dropout=dropout)

    def forward(self, x, skip):
        x = self.up_conv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.single_conv(x)
        return x


@register_model("UNetSimpleEncoderDecoder")
class UNetSimpleEncoderDecoder(nn.Module):
    """
    UNet variant with SingleConv blocks in both encoder and decoder while
    keeping the bottleneck and skip connections identical to the baseline.
    """

    def __init__(self, config: dict):
        super().__init__()

        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        depths = config["depths"]  # e.g., [32, 64, 128, 256, 512]
        dropout = config.get("dropout", 0.0)

        # Encoder - simplified
        self.down_conv_1 = SimpleDownSampling(in_channels, depths[0], dropout=dropout)
        self.down_conv_2 = SimpleDownSampling(depths[0], depths[1], dropout=dropout)
        self.down_conv_3 = SimpleDownSampling(depths[1], depths[2], dropout=dropout)
        self.down_conv_4 = SimpleDownSampling(depths[2], depths[3], dropout=dropout)

        # Bottleneck - unchanged
        self.bottleneck = Bottleneck(depths[3], depths[4], dropout=dropout)

        # Decoder - simplified
        self.up_conv_1 = SimpleUpSampling(depths[4], depths[3], dropout=dropout)
        self.up_conv_2 = SimpleUpSampling(depths[3], depths[2], dropout=dropout)
        self.up_conv_3 = SimpleUpSampling(depths[2], depths[1], dropout=dropout)
        self.up_conv_4 = SimpleUpSampling(depths[1], depths[0], dropout=dropout)

        # Output
        self.out_conv = nn.Conv2d(depths[0], out_channels, 1)

        # Final activation
        if config.get("final_activation") == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

    def forward(self, x):
        # Encoder
        single_conv_1, x = self.down_conv_1(x)
        single_conv_2, x = self.down_conv_2(x)
        single_conv_3, x = self.down_conv_3(x)
        single_conv_4, x = self.down_conv_4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.up_conv_1(x, single_conv_4)
        x = self.up_conv_2(x, single_conv_3)
        x = self.up_conv_3(x, single_conv_2)
        x = self.up_conv_4(x, single_conv_1)

        # Output
        out = self.out_conv(x)
        out = self.final_activation(out)

        return out

