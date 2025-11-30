"""Loss functions for segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss amplifies hard examples by raising the complement of the
    Tversky index to a power gamma (>1). Common choice: gamma = 4/3.
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.333,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        focal = (1 - tversky) ** self.gamma
        return focal


def create_loss(loss_config: dict) -> nn.Module:
    """Create loss function from configuration."""
    loss_type = loss_config['type']
    
    if loss_type == 'dice':
        return DiceLoss(smooth=loss_config.get('smooth', 1e-6))
    if loss_type == 'focal_tversky':
        return FocalTverskyLoss(
            alpha=loss_config.get('alpha', 0.3),
            beta=loss_config.get('beta', 0.7),
            gamma=loss_config.get('gamma', 1.333),
            smooth=loss_config.get('smooth', 1e-6)
        )
    raise ValueError(f"Unknown loss type: {loss_type}") 