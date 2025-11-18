"""UNet architecture without skip connections."""

import torch
import torch.nn as nn
from ..registry import register_model
from ..blocks.conv_blocks import DownSampling, Bottleneck, DoubleConv

@register_model('UNetNoSkip')
class UNetNoSkip(nn.Module):
    """
    UNet without skip connections - decoder does not use encoder features.
    
    This removes all skip connections, forcing the decoder to reconstruct
    from the bottleneck features alone. This tests the importance of
    skip connections for preserving spatial details.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config['depths']  # e.g., [32, 64, 128, 256, 512]
        dropout = config.get('dropout', 0.0)
        
        # Encoder - unchanged
        self.down_conv_1 = DownSampling(in_channels, depths[0], dropout=dropout)
        self.down_conv_2 = DownSampling(depths[0], depths[1], dropout=dropout)
        self.down_conv_3 = DownSampling(depths[1], depths[2], dropout=dropout)
        self.down_conv_4 = DownSampling(depths[2], depths[3], dropout=dropout)
        
        # Bottleneck - unchanged
        self.bottleneck = Bottleneck(depths[3], depths[4], dropout=dropout)
        
        # Decoder - manual implementation without skip connections
        # After bottleneck: x has depths[4] channels at 32x32
        
        # Upsampling layer 1: 32x32 -> 64x64
        self.up_1 = nn.ConvTranspose2d(depths[4], depths[3], kernel_size=2, stride=2)
        self.conv_1 = DoubleConv(depths[3], depths[3], dropout=dropout)  # no concatenation
        
        # Upsampling layer 2: 64x64 -> 128x128
        self.up_2 = nn.ConvTranspose2d(depths[3], depths[2], kernel_size=2, stride=2)
        self.conv_2 = DoubleConv(depths[2], depths[2], dropout=dropout)  # no concatenation
        
        # Upsampling layer 3: 128x128 -> 256x256
        self.up_3 = nn.ConvTranspose2d(depths[2], depths[1], kernel_size=2, stride=2)
        self.conv_3 = DoubleConv(depths[1], depths[1], dropout=dropout)  # no concatenation
        
        # Upsampling layer 4: 256x256 -> 512x512
        self.up_4 = nn.ConvTranspose2d(depths[1], depths[0], kernel_size=2, stride=2)
        self.conv_4 = DoubleConv(depths[0], depths[0], dropout=dropout)  # no concatenation
        
        # Output
        self.out_conv = nn.Conv2d(depths[0], out_channels, 1)
        
        # Final activation
        if config.get('final_activation') == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()
    
    def forward(self, x):
        # Encoder - we still need to run it for consistency, but skip features are not used
        _, x = self.down_conv_1(x)
        _, x = self.down_conv_2(x)
        _, x = self.down_conv_3(x)
        _, x = self.down_conv_4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder without skip connections
        x = self.up_1(x)  # 32x32 -> 64x64
        x = self.conv_1(x)
        
        x = self.up_2(x)  # 64x64 -> 128x128
        x = self.conv_2(x)
        
        x = self.up_3(x)  # 128x128 -> 256x256
        x = self.conv_3(x)
        
        x = self.up_4(x)  # 256x256 -> 512x512
        x = self.conv_4(x)
        
        # Output
        out = self.out_conv(x)
        out = self.final_activation(out)
        
        return out

