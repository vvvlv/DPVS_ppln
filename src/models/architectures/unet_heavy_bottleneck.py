"""UNet architecture with heavy bottleneck processing."""

import torch
import torch.nn as nn
from ..registry import register_model
from ..blocks.conv_blocks import DownSampling, UpSampling, DoubleConv

class HeavyBottleneck(nn.Module):
    """
    Heavy bottleneck with multiple processing layers.
    
    Instead of just one DoubleConv, this bottleneck has multiple
    sequential DoubleConv blocks to allow more complex transformations
    at the bottleneck level.
    """
    
    def __init__(self, in_channels: int, out_channels: int, depth: int = 3):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            depth: Number of DoubleConv blocks (default: 3)
        """
        super().__init__()
        
        # First expansion
        self.conv_in = DoubleConv(in_channels, out_channels)
        
        # Middle processing layers (same number of channels)
        self.middle_layers = nn.ModuleList([
            DoubleConv(out_channels, out_channels)
            for _ in range(depth - 1)
        ])
    
    def forward(self, x):
        x = self.conv_in(x)
        
        # Apply middle layers with residual connections
        for layer in self.middle_layers:
            identity = x
            x = layer(x)
            x = x + identity  # Residual connection
        
        return x


@register_model('UNetHeavyBottleneck')
class UNetHeavyBottleneck(nn.Module):
    """
    UNet with heavy bottleneck processing.
    
    Instead of adding more downsampling levels, this model keeps the same
    spatial resolution (32x32 at bottleneck) but adds multiple processing
    layers at the bottleneck with residual connections for better gradient flow.
    
    This allows the bottleneck to learn more complex transformations without
    losing too much spatial information.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config['depths']  # e.g., [32, 64, 128, 256, 512]
        bottleneck_depth = config.get('bottleneck_depth', 3)  # Number of processing layers
        
        # Encoder - 4 levels of downsampling (same as original UNet)
        self.down_conv_1 = DownSampling(in_channels, depths[0])
        self.down_conv_2 = DownSampling(depths[0], depths[1])
        self.down_conv_3 = DownSampling(depths[1], depths[2])
        self.down_conv_4 = DownSampling(depths[2], depths[3])
        
        # Heavy bottleneck with multiple processing layers
        self.bottleneck = HeavyBottleneck(depths[3], depths[4], depth=bottleneck_depth)
        
        # Decoder - 4 levels of upsampling
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
        double_conv_1, x = self.down_conv_1(x)  # 512x512 -> 256x256
        double_conv_2, x = self.down_conv_2(x)  # 256x256 -> 128x128
        double_conv_3, x = self.down_conv_3(x)  # 128x128 -> 64x64
        double_conv_4, x = self.down_conv_4(x)  # 64x64 -> 32x32
        
        # Heavy bottleneck with multiple processing layers
        x = self.bottleneck(x)  # 32x32 (deeper processing)
        
        # Decoder with skip connections
        x = self.up_conv_1(x, double_conv_4)  # 32x32 -> 64x64
        x = self.up_conv_2(x, double_conv_3)  # 64x64 -> 128x128
        x = self.up_conv_3(x, double_conv_2)  # 128x128 -> 256x256
        x = self.up_conv_4(x, double_conv_1)  # 256x256 -> 512x512
        
        # Output
        out = self.out_conv(x)
        out = self.final_activation(out)
        
        return out

