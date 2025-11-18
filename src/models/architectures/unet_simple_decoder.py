"""UNet architecture with simplified decoder (SingleConv instead of DoubleConv)."""

import torch
import torch.nn as nn
from ..registry import register_model
from ..blocks.conv_blocks import DownSampling, SingleConv

class SimpleUpSampling(nn.Module):
    """Upsampling block with SingleConv instead of DoubleConv."""
    
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels=input_size, out_channels=output_size, kernel_size=2, stride=2)
        self.single_conv = SingleConv(input_size, output_size, dropout=dropout)

    def forward(self, x, skip):
        x = self.up_conv(x) 
        x = torch.cat([x, skip], dim=1)
        x = self.single_conv(x)
        return x


@register_model('UNetSimpleDecoder')
class UNetSimpleDecoder(nn.Module):
    """
    UNet with simplified decoder - uses SingleConv instead of DoubleConv in decoder.
    
    This reduces the number of parameters and computations in the decoder while
    keeping the encoder and bottleneck unchanged.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config['depths']  # e.g., [32, 64, 128, 256, 512]
        dropout = config.get('dropout', 0.0)
        
        # Encoder - unchanged (DoubleConv)
        self.down_conv_1 = DownSampling(in_channels, depths[0], dropout=dropout)
        self.down_conv_2 = DownSampling(depths[0], depths[1], dropout=dropout)
        self.down_conv_3 = DownSampling(depths[1], depths[2], dropout=dropout)
        self.down_conv_4 = DownSampling(depths[2], depths[3], dropout=dropout)
        
        # Bottleneck - unchanged (DoubleConv)
        from ..blocks.conv_blocks import Bottleneck
        self.bottleneck = Bottleneck(depths[3], depths[4], dropout=dropout)
        
        # Decoder - simplified with SingleConv
        self.up_conv_1 = SimpleUpSampling(depths[4], depths[3], dropout=dropout)
        self.up_conv_2 = SimpleUpSampling(depths[3], depths[2], dropout=dropout) 
        self.up_conv_3 = SimpleUpSampling(depths[2], depths[1], dropout=dropout)
        self.up_conv_4 = SimpleUpSampling(depths[1], depths[0], dropout=dropout)
        
        # Output
        self.out_conv = nn.Conv2d(depths[0], out_channels, 1)
        
        # Final activation
        if config.get('final_activation') == 'sigmoid':
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
        
        # Decoder with skip connections - simplified
        x = self.up_conv_1(x, double_conv_4)
        x = self.up_conv_2(x, double_conv_3)
        x = self.up_conv_3(x, double_conv_2)
        x = self.up_conv_4(x, double_conv_1)
        
        # Output
        out = self.out_conv(x)
        out = self.final_activation(out)
        
        return out

