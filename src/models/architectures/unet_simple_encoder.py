"""UNet architecture with simplified encoder (SingleConv instead of DoubleConv)."""

import torch
import torch.nn as nn
from ..registry import register_model
from ..blocks.conv_blocks import SingleConv, UpSampling, Bottleneck

class SimpleDownSampling(nn.Module):
    """Downsampling block with SingleConv instead of DoubleConv."""
    
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        super().__init__()
        self.single_conv = SingleConv(input_size, output_size, dropout=dropout)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        conv_x = self.single_conv(x)
        x = self.max_pool(conv_x)
        return conv_x, x


@register_model('UNetSimpleEncoder')
class UNetSimpleEncoder(nn.Module):
    """
    UNet with simplified encoder - uses SingleConv instead of DoubleConv in encoder.
    
    This reduces the number of parameters and computations in the encoder while
    keeping the bottleneck and decoder unchanged.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config['depths']  # e.g., [32, 64, 128, 256, 512]
        dropout = config.get('dropout', 0.0)
        
        # Encoder - simplified with SingleConv
        self.down_conv_1 = SimpleDownSampling(in_channels, depths[0], dropout=dropout)
        self.down_conv_2 = SimpleDownSampling(depths[0], depths[1], dropout=dropout)
        self.down_conv_3 = SimpleDownSampling(depths[1], depths[2], dropout=dropout)
        self.down_conv_4 = SimpleDownSampling(depths[2], depths[3], dropout=dropout)
        
        # Bottleneck - unchanged (DoubleConv)
        self.bottleneck = Bottleneck(depths[3], depths[4], dropout=dropout)
        
        # Decoder - unchanged (DoubleConv)
        self.up_conv_1 = UpSampling(depths[4], depths[3], dropout=dropout)
        self.up_conv_2 = UpSampling(depths[3], depths[2], dropout=dropout) 
        self.up_conv_3 = UpSampling(depths[2], depths[1], dropout=dropout)
        self.up_conv_4 = UpSampling(depths[1], depths[0], dropout=dropout)
        
        # Output
        self.out_conv = nn.Conv2d(depths[0], out_channels, 1)
        
        # Final activation
        if config.get('final_activation') == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()
    
    def forward(self, x):
        # Encoder - simplified
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

