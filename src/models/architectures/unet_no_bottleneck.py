"""UNet architecture without bottleneck layer."""

import torch
import torch.nn as nn
from ..registry import register_model
from ..blocks.conv_blocks import DownSampling, DoubleConv

@register_model('UNetNoBottleneck')
class UNetNoBottleneck(nn.Module):
    """UNet architecture without bottleneck - goes directly from encoder to decoder."""
    
    def __init__(self, config: dict):
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config['depths']  # e.g., [32, 64, 128, 256]
        
        # Encoder - 4 levels of downsampling (same as UNet)
        self.down_conv_1 = DownSampling(in_channels, depths[0])
        self.down_conv_2 = DownSampling(depths[0], depths[1])
        self.down_conv_3 = DownSampling(depths[1], depths[2])
        self.down_conv_4 = DownSampling(depths[2], depths[3])
        
        # No bottleneck - go directly to decoder
        
        # Decoder - manual implementation with correct channel dimensions
        # After down_conv_4: x has depths[3] channels at 32x32
        
        # Upsampling layer 1: 32x32 -> 64x64
        self.up_1 = nn.ConvTranspose2d(depths[3], depths[2], kernel_size=2, stride=2)
        self.conv_1 = DoubleConv(depths[3] + depths[2], depths[2])  # 256+128 -> 128
        
        # Upsampling layer 2: 64x64 -> 128x128
        self.up_2 = nn.ConvTranspose2d(depths[2], depths[1], kernel_size=2, stride=2)
        self.conv_2 = DoubleConv(depths[2] + depths[1], depths[1])  # 128+64 -> 64
        
        # Upsampling layer 3: 128x128 -> 256x256
        self.up_3 = nn.ConvTranspose2d(depths[1], depths[0], kernel_size=2, stride=2)
        self.conv_3 = DoubleConv(depths[1] + depths[0], depths[0])  # 64+32 -> 32
        
        # Upsampling layer 4: 256x256 -> 512x512
        self.up_4 = nn.ConvTranspose2d(depths[0], depths[0], kernel_size=2, stride=2)
        self.conv_4 = DoubleConv(depths[0] + depths[0], depths[0])  # 32+32 -> 32
        
        # Output
        self.out_conv = nn.Conv2d(depths[0], out_channels, 1)
        
        # Final activation
        if config.get('final_activation') == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()
    
    def forward(self, x):
        # Encoder - 4 downsampling layers
        double_conv_1, x = self.down_conv_1(x)  # 512x512 -> 256x256
        double_conv_2, x = self.down_conv_2(x)  # 256x256 -> 128x128
        double_conv_3, x = self.down_conv_3(x)  # 128x128 -> 64x64
        double_conv_4, x = self.down_conv_4(x)  # 64x64 -> 32x32
        
        # No bottleneck - use x directly from encoder
        
        # Decoder with skip connections - 4 upsampling layers
        x = self.up_1(x)  # 32x32 -> 64x64
        x = torch.cat([x, double_conv_4], dim=1)
        x = self.conv_1(x)
        
        x = self.up_2(x)  # 64x64 -> 128x128
        x = torch.cat([x, double_conv_3], dim=1)
        x = self.conv_2(x)
        
        x = self.up_3(x)  # 128x128 -> 256x256
        x = torch.cat([x, double_conv_2], dim=1)
        x = self.conv_3(x)
        
        x = self.up_4(x)  # 256x256 -> 512x512
        x = torch.cat([x, double_conv_1], dim=1)
        x = self.conv_4(x)
        
        # Output
        out = self.out_conv(x)
        out = self.final_activation(out)
        
        return out

