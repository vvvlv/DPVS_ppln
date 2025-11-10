"""TransUNet architecture implementation.

Simple hybrid CNN-Transformer model based on UNet with transformer bottleneck.
"""

import torch
import torch.nn as nn
from ..registry import register_model
from ..blocks.conv_blocks import DownSampling, UpSampling
from ..blocks.transformer_blocks import TransformerEncoder


class TransformerBottleneck(nn.Module):
    """
    Transformer-based bottleneck for TransUNet.
    
    Applies transformer blocks for global context modeling.
    Handles channel projection from input to output dimensions.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Project to higher dimension for transformer
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            dim=out_channels,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Final convolution
        self.proj_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: Feature map (B, in_channels, H, W)
        
        Returns:
            Transformed feature map (B, out_channels, H, W)
        """
        x = self.proj_in(x)
        x = self.transformer(x)
        x = self.proj_out(x)
        return x


@register_model('TransUNet')
class TransUNet(nn.Module):
    """
    TransUNet: UNet with Transformer bottleneck.
    
    Same architecture as UNet but replaces the standard bottleneck
    with a transformer for global context modeling.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config['depths']  # e.g., [32, 64, 128, 256, 512]
        
        # Transformer configuration
        transformer_depth = config.get('transformer_depth', 6)
        transformer_heads = config.get('transformer_heads', 8)
        transformer_mlp_ratio = config.get('transformer_mlp_ratio', 4.0)
        transformer_dropout = config.get('transformer_dropout', 0.1)
        
        # Encoder (same as UNet)
        self.down_conv_1 = DownSampling(in_channels, depths[0])
        self.down_conv_2 = DownSampling(depths[0], depths[1])
        self.down_conv_3 = DownSampling(depths[1], depths[2])
        self.down_conv_4 = DownSampling(depths[2], depths[3])
        
        # Bottleneck (transformer instead of conv)
        self.bottleneck = TransformerBottleneck(
            in_channels=depths[3],
            out_channels=depths[4],
            depth=transformer_depth,
            num_heads=transformer_heads,
            mlp_ratio=transformer_mlp_ratio,
            dropout=transformer_dropout
        )
        
        # Decoder (same as UNet)
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
        
        # Bottleneck (transformer)
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
