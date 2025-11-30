"""UNet architecture with configurable convolution kernel size."""

import torch
import torch.nn as nn
from ..registry import register_model


class DoubleConvKernel(nn.Module):
    """Double convolution block with configurable kernel size."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSamplingKernel(nn.Module):
    def __init__(self, input_size: int, output_size: int, kernel_size: int):
        super().__init__()
        self.double_conv = DoubleConvKernel(input_size, output_size, kernel_size)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        conv_x = self.double_conv(x)
        x = self.max_pool(conv_x)
        return conv_x, x


class UpSamplingKernel(nn.Module):
    def __init__(self, input_size: int, output_size: int, kernel_size: int):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels=input_size, out_channels=output_size, kernel_size=2, stride=2
        )
        self.double_conv = DoubleConvKernel(input_size, output_size, kernel_size)

    def forward(self, x, skip):
        x = self.up_conv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv(x)
        return x


@register_model("UNetLargeKernel")
class UNetLargeKernel(nn.Module):
    """UNet with configurable kernel size in encoder, bottleneck, and decoder."""

    def __init__(self, config: dict):
        super().__init__()

        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        depths = config["depths"]  # e.g., [32, 64, 128, 256, 512]
        kernel_size = config.get("kernel_size", 3)
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError("kernel_size must be an odd integer >= 3.")

        # Encoder
        self.down_conv_1 = DownSamplingKernel(in_channels, depths[0], kernel_size)
        self.down_conv_2 = DownSamplingKernel(depths[0], depths[1], kernel_size)
        self.down_conv_3 = DownSamplingKernel(depths[1], depths[2], kernel_size)
        self.down_conv_4 = DownSamplingKernel(depths[2], depths[3], kernel_size)

        # Bottleneck
        self.bottleneck = DoubleConvKernel(depths[3], depths[4], kernel_size)

        # Decoder
        self.up_conv_1 = UpSamplingKernel(depths[4], depths[3], kernel_size)
        self.up_conv_2 = UpSamplingKernel(depths[3], depths[2], kernel_size)
        self.up_conv_3 = UpSamplingKernel(depths[2], depths[1], kernel_size)
        self.up_conv_4 = UpSamplingKernel(depths[1], depths[0], kernel_size)

        # Output
        self.out_conv = nn.Conv2d(depths[0], out_channels, 1)

        # Final activation
        if config.get("final_activation") == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

    def forward(self, x):
        # Encoder
        double_conv_1, x = self.down_conv_1(x)
        double_conv_2, x = self.down_conv_2(x)
        double_conv_3, x = self.down_conv_3(x)
        double_conv_4, x = self.down_conv_4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.up_conv_1(x, double_conv_4)
        x = self.up_conv_2(x, double_conv_3)
        x = self.up_conv_3(x, double_conv_2)
        x = self.up_conv_4(x, double_conv_1)

        # Output
        out = self.out_conv(x)
        out = self.final_activation(out)

        return out
