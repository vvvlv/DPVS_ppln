"""Evaluation metrics for segmentation."""

import torch
import numpy as np
from typing import Dict, Callable
from sklearn.metrics import roc_auc_score


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


def auc_score(pred, target) -> float:
    """
    Calculate AUC (Area Under ROC Curve).
    
    Args:
        pred: Predicted probabilities (continuous values 0-1)
        target: Ground truth binary masks (0 or 1)
    
    Returns:
        AUC score (0-1, higher is better)
    """
    # Flatten predictions and targets
    pred_np = pred.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()
    
    # Check if we have both classes (needed for AUC calculation)
    if len(np.unique(target_np)) < 2:
        # If only one class present, return 1.0 (perfect score)
        return 1.0
    
    try:
        auc = roc_auc_score(target_np, pred_np)
        return float(auc)
    except Exception as e:
        # If AUC calculation fails, return 0.5 (random classifier)
        print(f"Warning: AUC calculation failed: {e}")
        return 0.5


METRICS = {
    'dice': dice_coefficient,
    'iou': iou_score,
    'auc': auc_score
}


def create_metrics(metric_names: list) -> Dict[str, Callable]:
    """Create metric functions from names."""
    metrics = {}
    for name in metric_names:
        if name not in METRICS:
            raise ValueError(f"Unknown metric: {name}. Available: {list(METRICS.keys())}")
        metrics[name] = METRICS[name]
    return metrics 