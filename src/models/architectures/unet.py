"""UNet architecture implementation."""

import torch
import torch.nn as nn
from ..registry import register_model
from ..blocks.conv_blocks import DoubleConv


@register_model('UNet')
class UNet(nn.Module):
    """Basic UNet architecture."""
    
    def __init__(self, config: dict):
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config['depths']  # e.g., [32, 64, 128, 256, 512]
        
        # Encoder
        self.encoder1 = DoubleConv(in_channels, depths[0])
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = DoubleConv(depths[0], depths[1])
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = DoubleConv(depths[1], depths[2])
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = DoubleConv(depths[2], depths[3])
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(depths[3], depths[4])
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(depths[4], depths[3], 2, stride=2)
        self.decoder4 = DoubleConv(depths[4], depths[3])
        
        self.upconv3 = nn.ConvTranspose2d(depths[3], depths[2], 2, stride=2)
        self.decoder3 = DoubleConv(depths[3], depths[2])
        
        self.upconv2 = nn.ConvTranspose2d(depths[2], depths[1], 2, stride=2)
        self.decoder2 = DoubleConv(depths[2], depths[1])
        
        self.upconv1 = nn.ConvTranspose2d(depths[1], depths[0], 2, stride=2)
        self.decoder1 = DoubleConv(depths[1], depths[0])
        
        # Output
        self.out_conv = nn.Conv2d(depths[0], out_channels, 1)
        
        # Final activation
        if config.get('final_activation') == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Output
        out = self.out_conv(dec1)
        out = self.final_activation(out)
        
        return out 