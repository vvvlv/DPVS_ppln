"""UNet architecture with limited skip connections (drops finest-resolution skip)."""

import torch
import torch.nn as nn
from ..registry import register_model
from ..blocks.conv_blocks import DownSampling, Bottleneck, DoubleConv

@register_model('UNetLimitedSkip')
class UNetLimitedSkip(nn.Module):
    """
    UNet variant that omits the highest-resolution skip connection.
    
    Keeps four downsampling levels like the baseline UNet, but during decoding
    the final upsampling stage does not concatenate with the earliest encoder
    feature map. This allows us to isolate the contribution of the final skip
    connection while keeping the rest of the architecture unchanged.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config['depths']  # e.g., [32, 64, 128, 256, 512]
        
        # Encoder (same as baseline UNet)
        self.down_conv_1 = DownSampling(in_channels, depths[0])
        self.down_conv_2 = DownSampling(depths[0], depths[1])
        self.down_conv_3 = DownSampling(depths[1], depths[2])
        self.down_conv_4 = DownSampling(depths[2], depths[3])
        
        # Bottleneck (DoubleConv)
        self.bottleneck = Bottleneck(depths[3], depths[4])
        
        # Decoder - first three stages with skip connections
        # After bottleneck: x has depths[4] channels, up_1 outputs depths[3] channels
        # Skip connection double_conv_4 has depths[3] channels
        self.up_1 = nn.ConvTranspose2d(depths[4], depths[3], kernel_size=2, stride=2)
        self.conv_1 = DoubleConv(depths[3] + depths[3], depths[3])  # 256+256 -> 256
        
        # After conv_1: x has depths[3] channels, up_2 outputs depths[2] channels
        # Skip connection double_conv_3 has depths[2] channels
        self.up_2 = nn.ConvTranspose2d(depths[3], depths[2], kernel_size=2, stride=2)
        self.conv_2 = DoubleConv(depths[2] + depths[2], depths[2])  # 128+128 -> 128
        
        # After conv_2: x has depths[2] channels, up_3 outputs depths[1] channels
        # Skip connection double_conv_2 has depths[1] channels
        self.up_3 = nn.ConvTranspose2d(depths[2], depths[1], kernel_size=2, stride=2)
        self.conv_3 = DoubleConv(depths[1] + depths[1], depths[1])  # 64+64 -> 64
        
        # Final upsampling without skip connection
        self.up_4 = nn.ConvTranspose2d(depths[1], depths[0], kernel_size=2, stride=2)
        self.conv_4 = DoubleConv(depths[0], depths[0])  # no concatenation
        
        # Output
        self.out_conv = nn.Conv2d(depths[0], out_channels, 1)
        
        if config.get('final_activation') == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()
    
    def forward(self, x):
        # Encoder
        double_conv_1, x = self.down_conv_1(x)  # 512 -> 256
        double_conv_2, x = self.down_conv_2(x)  # 256 -> 128
        double_conv_3, x = self.down_conv_3(x)  # 128 -> 64
        double_conv_4, x = self.down_conv_4(x)  # 64  -> 32
        
        # Bottleneck
        x = self.bottleneck(x)  # 32 -> 32
        
        # Decoder with limited skips
        x = self.up_1(x)
        x = torch.cat([x, double_conv_4], dim=1)
        x = self.conv_1(x)
        
        x = self.up_2(x)
        x = torch.cat([x, double_conv_3], dim=1)
        x = self.conv_2(x)
        
        x = self.up_3(x)
        x = torch.cat([x, double_conv_2], dim=1)
        x = self.conv_3(x)
        
        x = self.up_4(x)
        x = self.conv_4(x)  # no skip connection with double_conv_1
        
        out = self.out_conv(x)
        out = self.final_activation(out)
        return out
