from ..blocks.conv_blocks import (
    DownSampling,
    UpSampling,
    Bottleneck,
    AttentionUpSampling,
    ASPPBottleneck,
)
import torch.nn as nn
from ..registry import register_model

@register_model('ASPP Attention UNet')
class ASPPAttUNet(nn.Module):
    """Basic UNet architecture with optional Attention Gates and ASPP bottleneck."""
    
    def __init__(self, config: dict):
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config['depths']  # e.g., [32, 64, 128, 256, 512]
        dropout = config.get('dropout', 0.1)
        
        # Encoder
        self.down_conv_1 = DownSampling(in_channels, depths[0], dropout=dropout)
        self.down_conv_2 = DownSampling(depths[0], depths[1], dropout=dropout)
        self.down_conv_3 = DownSampling(depths[1], depths[2], dropout=dropout)
        self.down_conv_4 = DownSampling(depths[2], depths[3], dropout=dropout)
        
        # Bottleneck (plain or ASPP-enhanced)
        self.bottleneck = ASPPBottleneck(depths[3], depths[4], dropout=dropout)
        
        # Decoder
        # use attention-gated upsampling (needs skip channel sizes)
        self.up_conv_1 = AttentionUpSampling(
            input_size=depths[4],
            skip_size=depths[3],
            output_size=depths[3],
            dropout=dropout
        )
        self.up_conv_2 = AttentionUpSampling(
            input_size=depths[3],
            skip_size=depths[2],
            output_size=depths[2],
            dropout=dropout
        )
        self.up_conv_3 = AttentionUpSampling(
            input_size=depths[2],
            skip_size=depths[1],
            output_size=depths[1],
            dropout=dropout
        )
        self.up_conv_4 = AttentionUpSampling(
            input_size=depths[1],
            skip_size=depths[0],
            output_size=depths[0],
            dropout=dropout
        )
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