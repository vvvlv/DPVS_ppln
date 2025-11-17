"""UNet architecture with shallow bottleneck at 64x64 resolution."""

import torch
import torch.nn as nn
from ..registry import register_model
from ..blocks.conv_blocks import DownSampling, Bottleneck, DoubleConv

@register_model('UNetShallow')
class UNetShallow(nn.Module):
    """
    UNet with 3 downsampling levels for bottleneck at 64x64 resolution.
    
    This allows the bottleneck to operate at a higher resolution (64x64)
    compared to the standard UNet (32x32), preserving more spatial detail
    while still capturing global context.
    
    Architecture:
    - Input: 512x512
    - After 3 downsamplings: 64x64 (bottleneck level)
    - Bottleneck processes at this resolution
    - 3 upsamplings back to 512x512
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config['depths']  # e.g., [32, 64, 128, 256]
        
        # Encoder - 3 levels of downsampling (512 -> 256 -> 128 -> 64)
        self.down_conv_1 = DownSampling(in_channels, depths[0])
        self.down_conv_2 = DownSampling(depths[0], depths[1])
        self.down_conv_3 = DownSampling(depths[1], depths[2])
        
        # Bottleneck at 64x64 resolution - higher resolution than standard UNet
        self.bottleneck = Bottleneck(depths[2], depths[3])
        
        # Decoder - manual implementation with correct channel dimensions
        # After bottleneck: x has depths[3] channels at 64x64
        
        # Upsampling layer 1: 64x64 -> 128x128
        # After bottleneck: x has depths[3] channels, up_1 outputs depths[2] channels
        # Skip connection double_conv_3 has depths[2] channels
        self.up_1 = nn.ConvTranspose2d(depths[3], depths[2], kernel_size=2, stride=2)
        self.conv_1 = DoubleConv(depths[2] + depths[2], depths[2])  # 128+128 -> 128
        
        # Upsampling layer 2: 128x128 -> 256x256
        # After conv_1: x has depths[2] channels, up_2 outputs depths[1] channels
        # Skip connection double_conv_2 has depths[1] channels
        self.up_2 = nn.ConvTranspose2d(depths[2], depths[1], kernel_size=2, stride=2)
        self.conv_2 = DoubleConv(depths[1] + depths[1], depths[1])  # 64+64 -> 64
        
        # Upsampling layer 3: 256x256 -> 512x512
        # After conv_2: x has depths[1] channels, up_3 outputs depths[0] channels
        # Skip connection double_conv_1 has depths[0] channels
        self.up_3 = nn.ConvTranspose2d(depths[1], depths[0], kernel_size=2, stride=2)
        self.conv_3 = DoubleConv(depths[0] + depths[0], depths[0])  # 32+32 -> 32
        
        # Output
        self.out_conv = nn.Conv2d(depths[0], out_channels, 1)
        
        # Final activation
        if config.get('final_activation') == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()
    
    def forward(self, x):
        # Encoder - 3 downsampling levels
        double_conv_1, x = self.down_conv_1(x)  # 512x512 -> 256x256
        double_conv_2, x = self.down_conv_2(x)  # 256x256 -> 128x128
        double_conv_3, x = self.down_conv_3(x)  # 128x128 -> 64x64
        
        # Bottleneck at 64x64 - captures context at higher resolution
        x = self.bottleneck(x)
        
        # Decoder with skip connections - 3 upsampling levels
        x = self.up_1(x)  # 64x64 -> 128x128
        x = torch.cat([x, double_conv_3], dim=1)
        x = self.conv_1(x)
        
        x = self.up_2(x)  # 128x128 -> 256x256
        x = torch.cat([x, double_conv_2], dim=1)
        x = self.conv_2(x)
        
        x = self.up_3(x)  # 256x256 -> 512x512
        x = torch.cat([x, double_conv_1], dim=1)
        x = self.conv_3(x)
        
        # Output
        out = self.out_conv(x)
        out = self.final_activation(out)
        
        return out

