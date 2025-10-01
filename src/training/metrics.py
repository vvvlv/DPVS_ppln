"""Evaluation metrics for segmentation."""

import torch
from typing import Dict, Callable


def dice_coefficient(pred, target, smooth: float = 1e-6) -> float:
    """Calculate Dice coefficient."""
    pred = (pred > 0.5).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def iou_score(pred, target, smooth: float = 1e-6) -> float:
    """Calculate IoU (Jaccard index)."""
    pred = (pred > 0.5).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


METRICS = {
    'dice': dice_coefficient,
    'iou': iou_score
}


def create_metrics(metric_names: list) -> Dict[str, Callable]:
    """Create metric functions from names."""
    metrics = {}
    for name in metric_names:
        if name not in METRICS:
            raise ValueError(f"Unknown metric: {name}. Available: {list(METRICS.keys())}")
        metrics[name] = METRICS[name]
    return metrics 