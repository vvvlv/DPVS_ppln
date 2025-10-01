"""Loss functions for segmentation."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        return 1 - dice


def create_loss(loss_config: dict) -> nn.Module:
    """Create loss function from configuration."""
    loss_type = loss_config['type']
    
    if loss_type == 'dice':
        return DiceLoss(smooth=loss_config.get('smooth', 1e-6))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}") 