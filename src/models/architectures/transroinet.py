"""TransRoiNet architecture: RoiNet with Transformer in bottleneck."""

import torch
import torch.nn as nn
from ..registry import register_model
from ..blocks.conv_blocks import ResidualBlock
from ..blocks.transformer_blocks import TransformerEncoder


@register_model('TransRoiNet')
class TransRoiNet(nn.Module):
    """
    TransRoiNet: RoiNet architecture with Transformer-enhanced bottleneck.
    
    Combines the strengths of both architectures:
    - RoiNet's residual blocks for robust gradient flow
    - RoiNet's deepened bottleneck for rich feature extraction
    - Transformer for capturing long-range spatial dependencies
    - Skip connections preserve fine details
    
    Architecture:
        Encoder (Residual CNN):
            - conv0: Full resolution (ch_in -> depths[0])
            - conv1 + pool1: Downsample to 1/2
            - conv2 + pool2: Downsample to 1/4
        
        Bottleneck (Residual + Transformer):
            - bottle1: First residual processing
            - transformer: Global context via self-attention
            - bottle2: Second residual processing
            - merge2: Combine with skip connection
        
        Decoder (Residual CNN):
            - conv3 + up3: Upsample to 1/2, merge with skip1
            - conv4 + up4: Upsample to full, merge with skip0
            - conv5: Final refinement
    
    Key benefits:
    - Residual connections for better training
    - Transformer captures global vessel patterns
    - Large receptive field (k_size=9)
    - Proven RoiNet structure + attention mechanism
    """
    
    def __init__(self, config: dict):
        """
        Initialize TransRoiNet from configuration.
        
        Args:
            config: Model configuration dictionary with:
                - in_channels: Input channels (typically 3 for RGB)
                - out_channels: Output channels (typically 1 for binary segmentation)
                - depths: List of channel sizes [32, 64, 128, 128, 64, 32]
                - kernel_size: Kernel size for residual blocks (default: 9)
                - transformer_depth: Number of transformer blocks (default: 2)
                - transformer_heads: Number of attention heads (default: 8)
                - transformer_mlp_ratio: MLP hidden dim ratio (default: 4.0)
                - transformer_dropout: Dropout in transformer (default: 0.1)
                - final_activation: 'sigmoid' or 'none' (default: 'sigmoid')
        """
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config.get('depths', [32, 64, 128, 128, 64, 32])
        k_size = config.get('kernel_size', 9)
        final_activation = config.get('final_activation', 'sigmoid')
        
        # Transformer config
        transformer_depth = config.get('transformer_depth', 2)
        transformer_heads = config.get('transformer_heads', 8)
        transformer_mlp_ratio = config.get('transformer_mlp_ratio', 4.0)
        transformer_dropout = config.get('transformer_dropout', 0.1)
        
        # Validate depths
        assert len(depths) == 6, "TransRoiNet requires exactly 6 depth values"
        
        self.depths = depths
        self.dict_module = nn.ModuleDict()
        
        # ------------------ Encoder ------------------
        # Block 0: Full resolution features
        self.dict_module.add_module(
            "conv0", 
            ResidualBlock(in_channels, depths[0], k_size=k_size)
        )
        
        # Block 1: Downsample once
        self.dict_module.add_module(
            "conv1",
            ResidualBlock(depths[0], depths[1], k_size=k_size)
        )
        # Downsample & double channels
        self.dict_module.add_module(
            "pool1",
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(depths[1], depths[1] * 2, kernel_size=1)
            )
        )
        
        # Block 2: Further encoding
        self.dict_module.add_module(
            "conv2",
            ResidualBlock(depths[1] * 2, depths[2], k_size=k_size)
        )
        # Downsample & double channels
        self.dict_module.add_module(
            "pool2",
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(depths[2], depths[2] * 2, kernel_size=1)
            )
        )
        
        # ------------------ Bottleneck (Residual + Transformer) ------------------
        # First residual block
        self.dict_module.add_module(
            "bottle1",
            ResidualBlock(depths[2] * 2, depths[2] * 2, k_size=k_size)
        )
        
        # Transformer for global context
        self.dict_module.add_module(
            "transformer",
            TransformerEncoder(
                dim=depths[2] * 2,
                depth=transformer_depth,
                num_heads=transformer_heads,
                mlp_ratio=transformer_mlp_ratio,
                dropout=transformer_dropout
            )
        )
        
        # Second residual block after transformer
        self.dict_module.add_module(
            "bottle2",
            ResidualBlock(depths[2] * 2, depths[2] * 2, k_size=k_size)
        )
        
        # Merge skip2 (from encoder) with the deepened bottleneck output
        self.dict_module.add_module(
            "merge2",
            nn.Sequential(
                nn.Conv2d(depths[2] * 4, depths[2] * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(depths[2] * 2),
                nn.ReLU(inplace=True)
            )
        )
        
        # ------------------ Decoder ------------------
        # Block 3: Upsample from bottleneck
        self.dict_module.add_module(
            "conv3",
            ResidualBlock(depths[2] * 2, depths[3], k_size=k_size)
        )
        self.dict_module.add_module(
            "up3",
            nn.Sequential(
                nn.ConvTranspose2d(depths[3], depths[3] // 2, kernel_size=2, stride=2)
            )
        )
        # Merge with skip connection from Block 1
        self.dict_module.add_module(
            "merge3",
            nn.Sequential(
                nn.Conv2d((depths[3] // 2) + (depths[1] * 2), depths[1], 
                         kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(depths[1]),
                nn.ReLU(inplace=True)
            )
        )
        
        # Block 4: Further upsampling
        self.dict_module.add_module(
            "conv4",
            ResidualBlock(depths[1], depths[4], k_size=k_size)
        )
        self.dict_module.add_module(
            "up4",
            nn.Sequential(
                nn.ConvTranspose2d(depths[4], depths[4] // 2, kernel_size=2, stride=2)
            )
        )
        self.dict_module.add_module(
            "merge4",
            nn.Sequential(
                nn.Conv2d((depths[4] // 2) + depths[0], depths[0],
                         kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(depths[0]),
                nn.ReLU(inplace=True)
            )
        )
        
        # Block 5: Final refinement
        self.dict_module.add_module(
            "conv5",
            ResidualBlock(depths[0], depths[5], k_size=k_size)
        )
        
        # Final classification layer
        final_layers = [nn.Conv2d(depths[5], out_channels, kernel_size=1, bias=False)]
        if final_activation == 'sigmoid':
            final_layers.append(nn.Sigmoid())
        
        self.dict_module.add_module("final", nn.Sequential(*final_layers))
    
    def forward(self, x):
        """
        Forward pass through TransRoiNet.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output segmentation (B, out_channels, H, W)
        """
        # Encoder
        out0 = self.dict_module["conv0"](x)           # (B, depths[0], H, W) -> skip0
        
        out1 = self.dict_module["conv1"](out0)        # (B, depths[1], H, W)
        out1 = self.dict_module["pool1"](out1)        # (B, depths[1]*2, H/2, W/2) -> skip1
        skip1 = out1
        
        out2 = self.dict_module["conv2"](out1)        # (B, depths[2], H/2, W/2)
        out2 = self.dict_module["pool2"](out2)        # (B, depths[2]*2, H/4, W/4) -> skip2
        skip2 = out2
        
        # Bottleneck (Residual + Transformer + Residual)
        bottle1 = self.dict_module["bottle1"](out2)   # (B, depths[2]*2, H/4, W/4)
        
        # Apply transformer for global context
        trans_out = self.dict_module["transformer"](bottle1)  # (B, depths[2]*2, H/4, W/4)
        
        bottle2 = self.dict_module["bottle2"](trans_out)  # (B, depths[2]*2, H/4, W/4)
        
        # Merge the original skip2 with the deepened features
        bottle_cat = torch.cat([bottle2, skip2], dim=1)  # (B, depths[2]*4, H/4, W/4)
        bottle_out = self.dict_module["merge2"](bottle_cat)  # (B, depths[2]*2, H/4, W/4)
        
        # Decoder
        out3 = self.dict_module["conv3"](bottle_out)  # (B, depths[3], H/4, W/4)
        out3 = self.dict_module["up3"](out3)          # (B, depths[3]//2, H/2, W/2)
        # Merge with skip1 (from pool1)
        out3 = torch.cat([out3, skip1], dim=1)        # (B, depths[3]//2 + depths[1]*2, H/2, W/2)
        out3 = self.dict_module["merge3"](out3)       # (B, depths[1], H/2, W/2)
        
        out4 = self.dict_module["conv4"](out3)        # (B, depths[4], H/2, W/2)
        out4 = self.dict_module["up4"](out4)          # (B, depths[4]//2, H, W)
        # Merge with skip0 (from conv0)
        out4 = torch.cat([out4, out0], dim=1)         # (B, depths[4]//2 + depths[0], H, W)
        out4 = self.dict_module["merge4"](out4)       # (B, depths[0], H, W)
        
        out5 = self.dict_module["conv5"](out4)        # (B, depths[5], H, W)
        final = self.dict_module["final"](out5)       # (B, out_channels, H, W)
        
        return final

