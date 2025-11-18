"""Convolutional building blocks."""

import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from torch import Tensor
import torch

class SingleConv(nn.Module):
    """Single convolution block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.single_conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.single_conv(x)


class DoubleConv(nn.Module):
    """Two consecutive convolution blocks."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.double_conv(x)
    

class DownSampling(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        self.double_conv = DoubleConv(
            input_size,
            output_size,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: Any):
        conv_x = self.double_conv(x)
        x = self.max_pool(conv_x)
        return conv_x, x
    
class UpSampling(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels=input_size, out_channels=output_size, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(
            input_size,
            output_size,
            kernel_size=kernel_size,
            dropout=dropout
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up_conv(x) 
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv(x)
        return x
    
class Bottleneck(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        self.bottleneck = DoubleConv(
            input_size,
            output_size,
            kernel_size=kernel_size,
            dropout=dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.bottleneck(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutions and skip connection.
    
    Used in RoiNet architecture. Includes batch normalization and ReLU activation.
    """
    
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 k_size: int = 3, dilation: int = 1):
        super().__init__()
        
        padding = k_size // 2 * dilation
        
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=k_size, stride=stride, 
            padding=padding, bias=False, dilation=dilation
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=k_size, stride=1,
            padding=padding, bias=False, dilation=dilation
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        
        return out 