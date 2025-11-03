"""UNet architecture implementation."""

import torch
import torch.nn as nn
from ..registry import register_model
from ..blocks.conv_blocks import DownSampling, UpSampling, Bottleneck

@register_model('UNet')
class UNet(nn.Module):
    """Basic UNet architecture."""
    
    def __init__(self, config: dict):
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config['depths']  # e.g., [32, 64, 128, 256, 512]
        
        # Encoder
        self.down_conv_1 = DownSampling(in_channels, depths[0])
        self.down_conv_2 = DownSampling(depths[0], depths[1])
        self.down_conv_3 = DownSampling(depths[1], depths[2])
        self.down_conv_4 = DownSampling(depths[2], depths[3])
        
        # Bottleneck
        self.bottleneck = Bottleneck(depths[3], depths[4])
        
        # Decoder
        self.up_conv_1 = UpSampling(depths[4], depths[3])
        self.up_conv_2 = UpSampling(depths[3], depths[2]) 
        self.up_conv_3 = UpSampling(depths[2], depths[1])
        self.up_conv_4 = UpSampling(depths[1], depths[0])
        
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
        
        # Decoder with skip connections
        x = self.up_conv_1(x, double_conv_4)
        x = self.up_conv_2(x, double_conv_3)
        x = self.up_conv_3(x, double_conv_2)
        x = self.up_conv_4(x, double_conv_1)
        
        # Output
        out = self.out_conv(x)
        out = self.final_activation(out)
        
        return out 