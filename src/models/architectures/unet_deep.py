"""UNet architecture with deeper bottleneck."""

import torch
import torch.nn as nn
from ..registry import register_model
from ..blocks.conv_blocks import DownSampling, Bottleneck, DoubleConv

@register_model('UNetDeep')
class UNetDeep(nn.Module):
    """
    UNet with 5 downsampling levels for deeper bottleneck at 16x16 resolution.
    
    This allows the bottleneck to operate at a lower resolution where it can
    more effectively capture global context.
    
    Architecture:
    - Input: 512x512
    - After 5 downsamplings: 16x16 (bottleneck level)
    - Bottleneck processes at this very low resolution
    - 5 upsamplings back to 512x512
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config['depths']  # e.g., [32, 64, 128, 256, 512, 1024]
        
        # Encoder - 5 levels of downsampling (512 -> 256 -> 128 -> 64 -> 32 -> 16)
        self.down_conv_1 = DownSampling(in_channels, depths[0])
        self.down_conv_2 = DownSampling(depths[0], depths[1])
        self.down_conv_3 = DownSampling(depths[1], depths[2])
        self.down_conv_4 = DownSampling(depths[2], depths[3])
        self.down_conv_5 = DownSampling(depths[3], depths[4])
        
        # Bottleneck at 16x16 resolution - this is where global context is captured
        self.bottleneck = Bottleneck(depths[4], depths[5])
        
        # Decoder - manual implementation with correct channel dimensions
        # After bottleneck: x has depths[5] channels at 16x16
        
        # Upsampling layer 1: 16x16 -> 32x32
        self.up_1 = nn.ConvTranspose2d(depths[5], depths[4], kernel_size=2, stride=2)
        self.conv_1 = DoubleConv(depths[4] + depths[4], depths[4])  # 512+512 -> 512
        
        # Upsampling layer 2: 32x32 -> 64x64
        # After conv_1: x has depths[4] channels, up_2 outputs depths[3] channels
        # Skip connection double_conv_4 has depths[3] channels
        self.up_2 = nn.ConvTranspose2d(depths[4], depths[3], kernel_size=2, stride=2)
        self.conv_2 = DoubleConv(depths[3] + depths[3], depths[3])  # 256+256 -> 256
        
        # Upsampling layer 3: 64x64 -> 128x128
        # After conv_2: x has depths[3] channels, up_3 outputs depths[2] channels
        # Skip connection double_conv_3 has depths[2] channels
        self.up_3 = nn.ConvTranspose2d(depths[3], depths[2], kernel_size=2, stride=2)
        self.conv_3 = DoubleConv(depths[2] + depths[2], depths[2])  # 128+128 -> 128
        
        # Upsampling layer 4: 128x128 -> 256x256
        # After conv_3: x has depths[2] channels, up_4 outputs depths[1] channels
        # Skip connection double_conv_2 has depths[1] channels
        self.up_4 = nn.ConvTranspose2d(depths[2], depths[1], kernel_size=2, stride=2)
        self.conv_4 = DoubleConv(depths[1] + depths[1], depths[1])  # 64+64 -> 64
        
        # Upsampling layer 5: 256x256 -> 512x512
        # After conv_4: x has depths[1] channels, up_5 outputs depths[0] channels
        # Skip connection double_conv_1 has depths[0] channels
        self.up_5 = nn.ConvTranspose2d(depths[1], depths[0], kernel_size=2, stride=2)
        self.conv_5 = DoubleConv(depths[0] + depths[0], depths[0])  # 32+32 -> 32
        
        # Output
        self.out_conv = nn.Conv2d(depths[0], out_channels, 1)
        
        # Final activation
        if config.get('final_activation') == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()
    
    def forward(self, x):
        # Encoder - 5 downsampling levels
        double_conv_1, x = self.down_conv_1(x)  # 512x512 -> 256x256
        double_conv_2, x = self.down_conv_2(x)  # 256x256 -> 128x128
        double_conv_3, x = self.down_conv_3(x)  # 128x128 -> 64x64
        double_conv_4, x = self.down_conv_4(x)  # 64x64 -> 32x32
        double_conv_5, x = self.down_conv_5(x)  # 32x32 -> 16x16
        
        # Bottleneck at 16x16 - captures global context
        x = self.bottleneck(x)
        
        # Decoder with skip connections - 5 upsampling levels
        x = self.up_1(x)  # 16x16 -> 32x32
        x = torch.cat([x, double_conv_5], dim=1)
        x = self.conv_1(x)
        
        x = self.up_2(x)  # 32x32 -> 64x64
        x = torch.cat([x, double_conv_4], dim=1)
        x = self.conv_2(x)
        
        x = self.up_3(x)  # 64x64 -> 128x128
        x = torch.cat([x, double_conv_3], dim=1)
        x = self.conv_3(x)
        
        x = self.up_4(x)  # 128x128 -> 256x256
        x = torch.cat([x, double_conv_2], dim=1)
        x = self.conv_4(x)
        
        x = self.up_5(x)  # 256x256 -> 512x512
        x = torch.cat([x, double_conv_1], dim=1)
        x = self.conv_5(x)
        
        # Output
        out = self.out_conv(x)
        out = self.final_activation(out)
        
        return out

