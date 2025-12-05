import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from torch import Tensor
import torch
from torchvision.models.segmentation.deeplabv3 import ASPP
from ..blocks.conv_blocks import DoubleConv


class AttentionGate(nn.Module):
    """
    The implementation of Attention Gate is from my bachelor thesis Maria Matusiskova.
    Attention gate for skip connections (from thesis mm_network.py).

    F_g: num of channels in gating signal g (decoder feature)
    F_l: num of channels in local feature map x (skip feature)
    F_int: num of intermediate channels for attention computation
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        """
        x: skip connection feature (local)
        g: decoder feature (gating)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    
class AttentionUpSampling(nn.Module):
    """
    UpSampling with Attention Gate on the skip connection.

    input_size:  channels of decoder input
    skip_size:   channels of skip connection
    output_size: channels after upsampling & fusion
    """
    def __init__(
        self,
        input_size: int,
        skip_size: int,
        output_size: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=2,
                stride=2,
            )
        
        self.attention_gate = AttentionGate(      
            F_g=output_size,     
            F_l=skip_size,     
            F_int=output_size // 2, 
        )
        self.double_conv = DoubleConv(
            output_size + skip_size, 
            output_size,
            kernel_size=kernel_size,
            dropout=dropout
        )

    def forward(self, x: Tensor, skip: Tensor = None) -> Tensor:
        x = self.up_conv(x)
        
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        skip = self.attention_gate(skip, x) 

        x = torch.cat([skip, x], dim=1)
        x = self.double_conv(x)
        return x

class ASPPBottleneck(nn.Module):
    """
    Bottleneck with DoubleConv + ASPP, inspired by mm_network thesis.

    input_size:  channels from last encoder layer
    output_size: channels in bottleneck (and ASPP)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        atrous_rates=(6, 12, 18),
    ):
        super().__init__()
        self.double_conv = DoubleConv(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.aspp = ASPP(
            in_channels=output_size,
            atrous_rates=list(atrous_rates),
            out_channels=output_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.double_conv(x)
        x = self.aspp(x)
        return x