"""RoiNet architecture implementation."""

import torch
import torch.nn as nn
from ..registry import register_model
from ..blocks.conv_blocks import ResidualBlock


@register_model('RoiNet')
class RoiNet(nn.Module):
    """
    RoiNet: A U-Net style architecture with residual blocks and deepened bottleneck.
    
    Key features:
    - Encoder-decoder structure with skip connections
    - Residual blocks for feature extraction
    - Deepened bottleneck with multiple residual blocks
    - Three resolution levels (full, 1/2, 1/4)
    - Configurable kernel size for residual blocks
    
    Architecture:
        Encoder:
            - conv0: Full resolution (ch_in -> depths[0])
            - conv1 + pool1: Downsample to 1/2 (depths[1] -> depths[1]*2)
            - conv2 + pool2: Downsample to 1/4 (depths[2] -> depths[2]*2)
        Bottleneck (deepened):
            - bottle1, bottle2: Extra residual blocks
            - Merge with skip2
        Decoder:
            - conv3 + up3: Upsample to 1/2, merge with skip1
            - conv4 + up4: Upsample to full, merge with skip0
            - conv5: Final refinement
            - final: Output layer with sigmoid
    """
    
    def __init__(self, config: dict):
        """
        Initialize RoiNet from configuration.
        
        Args:
            config: Model configuration dictionary with:
                - in_channels: Input channels (typically 3 for RGB)
                - out_channels: Output channels (typically 1 for binary segmentation)
                - depths: List of channel sizes [32, 64, 128, 128, 64, 32]
                - kernel_size: Kernel size for residual blocks (default: 9)
                - final_activation: 'sigmoid' or 'none' (default: 'sigmoid')
        """
        super().__init__()
        
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        depths = config.get('depths', [32, 64, 128, 128, 64, 32])
        k_size = config.get('kernel_size', 9)
        final_activation = config.get('final_activation', 'sigmoid')
        
        # Validate depths
        assert len(depths) == 6, "RoiNet requires exactly 6 depth values"
        
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
        
        # ------------------ Bottleneck (Deepened) ------------------
        # Add extra blocks to deepen the bottleneck
        self.dict_module.add_module(
            "bottle1",
            ResidualBlock(depths[2] * 2, depths[2] * 2, k_size=k_size)
        )
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
        Forward pass through RoiNet.
        
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
        
        # Bottleneck (deepened)
        bottle1 = self.dict_module["bottle1"](out2)   # (B, depths[2]*2, H/4, W/4)
        bottle2 = self.dict_module["bottle2"](bottle1) # (B, depths[2]*2, H/4, W/4)
        # Merge the original skip2 with the deepened features
        bottle_cat = torch.cat([bottle2, skip2], dim=1) # (B, depths[2]*4, H/4, W/4)
        bottle_out = self.dict_module["merge2"](bottle_cat) # (B, depths[2]*2, H/4, W/4)
        
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
