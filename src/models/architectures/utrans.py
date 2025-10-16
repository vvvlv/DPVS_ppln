"""UTrans architecture: UNet with Transformer bottleneck."""

import torch
import torch.nn as nn
from ..registry import register_model
from ..blocks.conv_blocks import DoubleConv
from ..blocks.transformer_blocks import TransformerEncoder


@register_model('UTrans')
class UTrans(nn.Module):
    """
    UTrans: UNet with Transformer bottleneck.
    
    Combines the best of both worlds:
    - CNN encoder/decoder for local feature extraction and spatial details
    - Transformer bottleneck for capturing long-range dependencies
    
    Architecture:
        Encoder (CNN):
            - 4 levels of DoubleConv + MaxPool
            - Progressive downsampling: H -> H/2 -> H/4 -> H/8 -> H/16
        
        Bottleneck (Transformer):
            - Stack of transformer blocks operating on H/16 resolution
            - Captures global context and long-range dependencies
        
        Decoder (CNN):
            - 4 levels of Upsample + DoubleConv with skip connections
            - Progressive upsampling back to original resolution
    
    Key benefits:
    - Local features from CNN paths
    - Global context from transformer
    - Skip connections preserve fine details
    - Extensible: easy to add transformers at other levels
    """
    
    def __init__(self, config: dict):
        """
        Initialize UTrans from configuration.
        
        Args:
            config: Model configuration dictionary with:
                - in_channels: Input channels (typically 3 for RGB)
                - out_channels: Output channels (typically 1 for binary segmentation)
                - depths: List of channel sizes [64, 128, 256, 512, 1024]
                - transformer_depth: Number of transformer blocks (default: 4)
                - transformer_heads: Number of attention heads (default: 8)
                - transformer_mlp_ratio: MLP hidden dim ratio (default: 4.0)
                - transformer_dropout: Dropout in transformer (default: 0.1)
                - final_activation: 'sigmoid' or 'none' (default: 'sigmoid')
        """
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config.get('depths', [64, 128, 256, 512, 1024])
        
        # Transformer config
        transformer_depth = config.get('transformer_depth', 4)
        transformer_heads = config.get('transformer_heads', 8)
        transformer_mlp_ratio = config.get('transformer_mlp_ratio', 4.0)
        transformer_dropout = config.get('transformer_dropout', 0.1)
        
        # Validate depths
        assert len(depths) == 5, "UTrans requires exactly 5 depth values"
        
        # ------------------ Encoder (CNN) ------------------
        self.encoder1 = DoubleConv(in_channels, depths[0])
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = DoubleConv(depths[0], depths[1])
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = DoubleConv(depths[1], depths[2])
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = DoubleConv(depths[2], depths[3])
        self.pool4 = nn.MaxPool2d(2)
        
        # ------------------ Bottleneck (Transformer) ------------------
        # First, a conv to get to bottleneck channels
        self.bottleneck_conv_in = DoubleConv(depths[3], depths[4])
        
        # Transformer encoder for global context
        self.transformer = TransformerEncoder(
            dim=depths[4],
            depth=transformer_depth,
            num_heads=transformer_heads,
            mlp_ratio=transformer_mlp_ratio,
            dropout=transformer_dropout
        )
        
        # Conv after transformer to refine features
        self.bottleneck_conv_out = DoubleConv(depths[4], depths[4])
        
        # ------------------ Decoder (CNN) ------------------
        self.upconv4 = nn.ConvTranspose2d(depths[4], depths[3], 2, stride=2)
        self.decoder4 = DoubleConv(depths[4], depths[3])
        
        self.upconv3 = nn.ConvTranspose2d(depths[3], depths[2], 2, stride=2)
        self.decoder3 = DoubleConv(depths[3], depths[2])
        
        self.upconv2 = nn.ConvTranspose2d(depths[2], depths[1], 2, stride=2)
        self.decoder2 = DoubleConv(depths[2], depths[1])
        
        self.upconv1 = nn.ConvTranspose2d(depths[1], depths[0], 2, stride=2)
        self.decoder1 = DoubleConv(depths[1], depths[0])
        
        # ------------------ Output ------------------
        self.out_conv = nn.Conv2d(depths[0], out_channels, 1)
        
        # Final activation
        if config.get('final_activation') == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()
    
    def forward(self, x):
        """
        Forward pass through UTrans.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output segmentation (B, out_channels, H, W)
        """
        # Encoder path (CNN)
        enc1 = self.encoder1(x)           # (B, depths[0], H, W)
        enc2 = self.encoder2(self.pool1(enc1))  # (B, depths[1], H/2, W/2)
        enc3 = self.encoder3(self.pool2(enc2))  # (B, depths[2], H/4, W/4)
        enc4 = self.encoder4(self.pool3(enc3))  # (B, depths[3], H/8, W/8)
        
        # Bottleneck (Transformer)
        bottleneck = self.pool4(enc4)           # (B, depths[3], H/16, W/16)
        bottleneck = self.bottleneck_conv_in(bottleneck)  # (B, depths[4], H/16, W/16)
        
        # Apply transformer for global context
        bottleneck = self.transformer(bottleneck)  # (B, depths[4], H/16, W/16)
        
        bottleneck = self.bottleneck_conv_out(bottleneck)  # (B, depths[4], H/16, W/16)
        
        # Decoder path (CNN) with skip connections
        dec4 = self.upconv4(bottleneck)         # (B, depths[3], H/8, W/8)
        dec4 = torch.cat([dec4, enc4], dim=1)   # (B, depths[4], H/8, W/8)
        dec4 = self.decoder4(dec4)              # (B, depths[3], H/8, W/8)
        
        dec3 = self.upconv3(dec4)               # (B, depths[2], H/4, W/4)
        dec3 = torch.cat([dec3, enc3], dim=1)   # (B, depths[3], H/4, W/4)
        dec3 = self.decoder3(dec3)              # (B, depths[2], H/4, W/4)
        
        dec2 = self.upconv2(dec3)               # (B, depths[1], H/2, W/2)
        dec2 = torch.cat([dec2, enc2], dim=1)   # (B, depths[2], H/2, W/2)
        dec2 = self.decoder2(dec2)              # (B, depths[1], H/2, W/2)
        
        dec1 = self.upconv1(dec2)               # (B, depths[0], H, W)
        dec1 = torch.cat([dec1, enc1], dim=1)   # (B, depths[1], H, W)
        dec1 = self.decoder1(dec1)              # (B, depths[0], H, W)
        
        # Output
        out = self.out_conv(dec1)               # (B, out_channels, H, W)
        out = self.final_activation(out)
        
        return out

