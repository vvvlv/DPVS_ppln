# Deep Learning Pipeline Specification
## UNet Training Framework with Transformer Support

**Target Hardware:** NVIDIA T4 (16GB)  
**Target Dataset:** FIVES (Fundus Image Vessel Segmentation)  
**Primary Goal:** Clean, modular pipeline for vessel segmentation experiments

---

## Table of Contents
1. [Design Philosophy](#1-design-philosophy)
2. [Project Structure](#2-project-structure)
3. [Configuration System](#3-configuration-system)
4. [Dataset Module](#4-dataset-module)
5. [Model System](#5-model-system)
6. [Training Module](#6-training-module)
7. [Testing & Inference](#7-testing--inference)
8. [Utilities](#8-utilities)
9. [Command-Line Interface](#9-command-line-interface)
10. [Implementation Checklist](#10-implementation-checklist)

---

## 1. Design Philosophy

### 1.1 Core Principles

**Simplicity**
- Two configuration types only: datasets (static) and experiments (dynamic)
- Each experiment is self-contained in a single YAML file
- Clear separation between what data we use and how we train

**Modularity**
- Plug-and-play components
- Model registry for easy architecture additions
- Composable building blocks
- Clear interfaces between modules

**Extensibility**
- Easy to add new models via registry
- Easy to add new losses/metrics
- Easy to add new augmentations
- Well-defined extension points

**Reproducibility**
- Configuration saved with every experiment
- Random seed management
- All hyperparameters explicit in experiment config

### 1.2 Configuration Strategy

**Two-Level Configuration:**

```
Dataset YAML
├── Static information only
├── Paths to data
├── Image properties (size, channels)
└── Normalization statistics

Experiment YAML
├── References a dataset
├── All training parameters (batch size, optimizer, etc.)
├── Model architecture specification
├── Data augmentation configuration
├── Logging and checkpointing settings
└── Complete and self-contained
```

**Why this works:**
- Dataset configs rarely change (just paths and basic properties)
- Experiment configs contain everything that varies
- Easy to compare experiments (diff two YAML files)
- Easy to share/reproduce (just share experiment YAML)
- No hidden defaults or inheritance complexity

---

## 2. Project Structure

```
project/
├── configs/
│   ├── datasets/                    # Static dataset definitions
│   │   ├── fives_512.yaml
│   │   ├── fives_1024.yaml
│   │   └── fives_original.yaml
│   │
│   └── experiments/                 # Complete experiment configurations
│       ├── exp001_baseline_unet.yaml
│       ├── exp002_unet_augmented.yaml
│       ├── exp003_focal_tversky.yaml
│       └── exp004_transunet.yaml
│
├── src/                             # Source code
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # Dataset class
│   │   └── transforms.py           # Augmentation pipeline
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py             # Model registration system
│   │   ├── blocks/                 # Building blocks
│   │   │   ├── __init__.py
│   │   │   ├── conv_blocks.py     # Convolution blocks
│   │   │   ├── attention.py       # Attention mechanisms
│   │   │   └── transformer.py     # Transformer blocks
│   │   └── architectures/
│   │       ├── __init__.py
│   │       └── unet.py            # UNet implementation
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Main training loop
│   │   ├── losses.py              # Loss functions
│   │   └── metrics.py             # Evaluation metrics
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py           # Inference engine
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # Config loading
│       ├── logger.py              # TensorBoard logging
│       └── helpers.py             # Utilities
│
├── scripts/                         # Executable scripts
│   ├── train.py                    # Main training script
│   ├── test.py                     # Testing script
│   └── infer.py                    # Inference script
│
├── notebooks/                       # Exploration notebooks
│   └── explore_data.ipynb
│
├── outputs/                         # Training outputs (gitignored)
│   └── experiments/
│       └── exp001_baseline_unet/
│           ├── config.yaml         # Saved experiment config
│           ├── checkpoints/        # Model checkpoints
│           ├── logs/               # TensorBoard logs
│           ├── predictions/        # Test predictions
│           └── metrics.json        # Final results
│
├── data/                           # Data directory (gitignored)
│   ├── FIVES512/
│   ├── FIVES1024/
│   └── FIVESoriginal/
│
├── train.sh                        # Training launcher
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 3. Configuration System

### 3.1 Dataset Configuration

**File:** `configs/datasets/fives_512.yaml`

**Purpose:** Define static dataset properties only.

```yaml
# Dataset: FIVES at 512x512 resolution
# Contains only immutable properties of the dataset

name: "FIVES512"
description: "Fundus Image Vessel Segmentation dataset at 512x512"

# Data locations
paths:
  root: "data/FIVES512"
  train: "data/FIVES512/train"
  val: "data/FIVES512/val"
  test: "data/FIVES512/test"

# Image properties (fixed for this dataset)
image_size: [512, 512]
num_channels: 3  # RGB
num_classes: 1   # Binary segmentation

# Normalization statistics (computed once)
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]
```

### 3.2 Experiment Configuration

**File:** `configs/experiments/exp001_baseline_unet.yaml`

**Purpose:** Complete experiment specification.

```yaml
# Experiment: Baseline UNet
# This file contains ALL configuration for this experiment

# ============================================================================
# METADATA
# ============================================================================
name: "exp001_baseline_unet"
description: "Baseline UNet with Dice loss, no augmentation"
tags: ["baseline", "unet", "dice"]

# ============================================================================
# DATASET
# ============================================================================
dataset: "configs/datasets/fives_512.yaml"

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================
data:
  batch_size: 8
  num_workers: 4
  pin_memory: true
  
  # Augmentation configuration
  augmentation:
    enabled: false
    # When enabled, specify:
    # geometric:
    #   enabled: true
    #   rotation_range: 15
    #   horizontal_flip: 0.5
    #   vertical_flip: 0.5
    # intensity:
    #   enabled: true
    #   brightness: 0.2
    #   contrast: 0.2
  
  # Preprocessing
  preprocessing:
    normalize: true
    pad_to_multiple: 32  # For UNet compatibility

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
model:
  type: "UNet"  # Registered model name
  
  # Architecture parameters
  in_channels: 3
  out_channels: 1
  depths: [32, 64, 128, 256, 512]  # Channel progression
  
  # Building blocks
  encoder_block: "conv"        # "conv", "residual"
  decoder_block: "conv"
  skip_connection: "concat"    # "concat", "add", "attention"
  
  # Output activation
  final_activation: "sigmoid"
  
  # Transformer bottleneck (optional)
  use_transformer: false
  # transformer:
  #   num_layers: 4
  #   num_heads: 8
  #   embed_dim: 512

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
training:
  epochs: 300
  
  # Optimizer
  optimizer:
    type: "adam"
    learning_rate: 0.0001
    weight_decay: 0.0001
  
  # Learning rate scheduler
  scheduler:
    type: "cosine"
    min_lr: 0.000001
    warmup_epochs: 5
  
  # Loss function
  loss:
    type: "dice"
    smooth: 0.000001
  
  # Metrics
  metrics:
    - "dice"
    - "iou"
    - "precision"
    - "recall"
  
  # Training features
  mixed_precision: true
  gradient_clip: 1.0
  
  # Validation
  validation:
    frequency: 1  # Every N epochs
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 50
    monitor: "val_dice"
    mode: "max"
  
  # Checkpointing
  checkpoint:
    save_best: true
    save_last: true
    save_top_k: 3
    monitor: "val_dice"
    mode: "max"

# ============================================================================
# LOGGING
# ============================================================================
logging:
  tensorboard: true
  log_images: true
  image_frequency: 5
  num_sample_images: 4

# ============================================================================
# OUTPUT
# ============================================================================
output:
  dir: "outputs/experiments/exp001_baseline_unet"
  save_predictions: true

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
seed: 42
device: "cuda"
```

### 3.3 Configuration Loading

**File:** `src/utils/config.py`

```python
"""Configuration loading and validation."""

from pathlib import Path
import yaml
from typing import Dict, Any

class Config:
    """
    Configuration loader.
    
    Responsibilities:
    - Load experiment config
    - Load referenced dataset config
    - Merge configurations
    - Validate required fields
    """
    
    def __init__(self, experiment_config_path: str):
        """
        Load experiment configuration.
        
        Args:
            experiment_config_path: Path to experiment YAML
        """
        pass
    
    def load(self) -> Dict[str, Any]:
        """
        Load and merge all configurations.
        
        Returns:
            Complete configuration dictionary
        """
        pass
    
    def validate(self) -> None:
        """Validate configuration has required fields."""
        pass
    
    def get(self, key: str, default=None) -> Any:
        """Get config value with dot notation (e.g., 'model.type')."""
        pass

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to experiment config
    
    Returns:
        Configuration dictionary
    """
    pass
```

---

## 4. Dataset Module

### 4.1 Dataset Class

**File:** `src/data/dataset.py`

```python
"""Dataset class for vessel segmentation."""

from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from typing import Dict, Optional, Tuple

class VesselSegmentationDataset(Dataset):
    """
    Dataset for vessel segmentation.
    
    Responsibilities:
    - Load images and masks from disk
    - Apply preprocessing (normalization, padding)
    - Apply augmentations (if provided)
    - Return data in correct format for training
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str,
        image_size: Tuple[int, int],
        mean: Tuple[float, ...],
        std: Tuple[float, ...],
        transform=None,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            root_dir: Dataset root directory
            split: "train", "val", or "test"
            image_size: Expected image size (H, W)
            mean: Normalization mean
            std: Normalization std
            transform: Augmentation pipeline (optional)
            normalize: Whether to normalize images
        """
        pass
    
    def __len__(self) -> int:
        """Return dataset size."""
        pass
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with keys:
                - "image": Image tensor (C, H, W)
                - "mask": Mask tensor (1, H, W)
                - "name": Image filename
        """
        pass
```

### 4.2 Data Augmentation

**File:** `src/data/transforms.py`

```python
"""Data augmentation pipeline using Albumentations."""

from typing import Optional

def get_transforms(augmentation_config: dict, is_training: bool = True):
    """
    Create augmentation pipeline from configuration.
    
    Args:
        augmentation_config: Augmentation section from experiment config
        is_training: Whether for training (apply augmentations) or not
    
    Returns:
        Albumentations Compose object or None
    
    Implementation notes:
    - Use Albumentations library
    - Build pipeline based on config flags
    - Apply geometric transforms (rotation, flip, scale)
    - Apply intensity transforms (brightness, contrast, gamma)
    - Apply noise transforms (gaussian noise)
    - Ensure masks are transformed consistently with images
    """
    pass
```

### 4.3 DataLoader Factory

**File:** `src/data/__init__.py`

```python
"""Data module initialization and factory functions."""

from torch.utils.data import DataLoader
from typing import Tuple

def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Experiment configuration dictionary
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Implementation notes:
    - Create datasets for train/val/test splits
    - Apply augmentation only to training data
    - Use batch_size, num_workers from config
    - Set shuffle=True for training, False for val/test
    - Use pin_memory for faster GPU transfer
    """
    pass
```

---

## 5. Model System

### 5.1 Model Registry

**File:** `src/models/registry.py`

```python
"""Model registration system for easy model management."""

from typing import Dict, Callable
import torch.nn as nn

# Global registry
_MODEL_REGISTRY: Dict[str, Callable] = {}

def register_model(name: str):
    """
    Decorator to register a model class.
    
    Usage:
        @register_model('UNet')
        class UNet(nn.Module):
            ...
    
    Args:
        name: Model name for registry
    """
    pass

def get_model(name: str, config: dict) -> nn.Module:
    """
    Get model instance from registry.
    
    Args:
        name: Registered model name
        config: Model configuration
    
    Returns:
        Model instance
    """
    pass

def list_models() -> list:
    """List all registered model names."""
    pass
```

### 5.2 Building Blocks

**File:** `src/models/blocks/conv_blocks.py`

```python
"""Convolutional building blocks."""

import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Basic convolution block: Conv → BN → ReLU.
    
    Used as the fundamental building block in the network.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, padding: int = 1):
        """Initialize convolution block."""
        pass
    
    def forward(self, x):
        """Forward pass."""
        pass

class DoubleConv(nn.Module):
    """
    Two consecutive convolution blocks.
    
    Standard building block for UNet encoder/decoder.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize double convolution block."""
        pass
    
    def forward(self, x):
        """Forward pass."""
        pass

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    
    For more advanced architectures.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize residual block."""
        pass
    
    def forward(self, x):
        """Forward pass."""
        pass
```

**File:** `src/models/blocks/attention.py`

```python
"""Attention mechanisms."""

import torch.nn as nn

class AttentionGate(nn.Module):
    """
    Attention gate for skip connections.
    
    Highlights important features from encoder to pass to decoder.
    """
    
    def __init__(self, gate_channels: int, skip_channels: int):
        """Initialize attention gate."""
        pass
    
    def forward(self, gate, skip):
        """
        Apply attention to skip connection.
        
        Args:
            gate: Features from decoder (lower resolution)
            skip: Features from encoder (higher resolution)
        
        Returns:
            Attended skip features
        """
        pass

class SpatialAttention(nn.Module):
    """
    Spatial attention module.
    
    Focuses on important spatial locations.
    """
    
    def __init__(self):
        """Initialize spatial attention."""
        pass
    
    def forward(self, x):
        """Apply spatial attention."""
        pass
```

**File:** `src/models/blocks/transformer.py`

```python
"""Transformer blocks for vision tasks."""

import torch.nn as nn

class TransformerBlock(nn.Module):
    """
    Vision transformer block.
    
    Self-attention + MLP with residual connections.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        """Initialize transformer block."""
        pass
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Token sequence (B, N, C)
        
        Returns:
            Transformed sequence (B, N, C)
        """
        pass

class TransformerBottleneck(nn.Module):
    """
    Transformer-based bottleneck for UNet.
    
    Replaces convolutional bottleneck with transformer layers.
    """
    
    def __init__(self, in_channels: int, num_layers: int = 4, 
                 num_heads: int = 8):
        """Initialize transformer bottleneck."""
        pass
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Spatial features (B, C, H, W)
        
        Returns:
            Transformed features (B, C, H, W)
        """
        pass
```

### 5.3 UNet Architecture

**File:** `src/models/architectures/unet.py`

```python
"""UNet architecture implementation."""

import torch
import torch.nn as nn
from ..registry import register_model

@register_model('UNet')
class UNet(nn.Module):
    """
    Flexible UNet architecture.
    
    Supports:
    - Configurable depth and channel sizes
    - Different building blocks (conv, residual)
    - Skip connection types (concat, add, attention)
    - Optional transformer bottleneck
    """
    
    def __init__(self, config: dict):
        """
        Initialize UNet from configuration.
        
        Args:
            config: Model configuration dictionary with:
                - in_channels: Input channels
                - out_channels: Output channels
                - depths: List of channel sizes at each level
                - encoder_block: Type of encoder block
                - decoder_block: Type of decoder block
                - skip_connection: Skip connection type
                - final_activation: Output activation
                - use_transformer: Whether to use transformer bottleneck
        """
        pass
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output segmentation (B, out_channels, H, W)
        """
        pass
```

### 5.4 Model Factory

**File:** `src/models/__init__.py`

```python
"""Model module initialization and factory."""

from .registry import register_model, get_model, list_models

def create_model(config: dict):
    """
    Create model from experiment configuration.
    
    Args:
        config: Experiment configuration
    
    Returns:
        Model instance
    
    Implementation notes:
    - Extract model config from experiment config
    - Get model type from config
    - Instantiate via registry
    - Initialize weights if specified
    """
    pass
```

---

## 6. Training Module

### 6.1 Trainer Class

**File:** `src/training/trainer.py`

```python
"""Main training loop handler."""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from typing import Dict

class Trainer:
    """
    Training loop manager.
    
    Responsibilities:
    - Execute training epochs
    - Execute validation epochs
    - Handle mixed precision training
    - Manage optimizer and scheduler
    - Handle checkpointing
    - Handle early stopping
    - Coordinate logging
    """
    
    def __init__(self, model: nn.Module, train_loader, val_loader, 
                 config: dict, device):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Experiment configuration
            device: Device to train on
        """
        pass
    
    def train(self):
        """
        Main training loop.
        
        Execute training for specified number of epochs,
        with validation, checkpointing, and early stopping.
        """
        pass
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        pass
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        pass
    
    def _create_optimizer(self):
        """Create optimizer from config."""
        pass
    
    def _create_scheduler(self):
        """Create learning rate scheduler from config."""
        pass
    
    def _create_loss(self):
        """Create loss function from config."""
        pass
    
    def _create_metrics(self):
        """Create metric functions from config."""
        pass
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        pass
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria met."""
        pass
```

### 6.2 Loss Functions

**File:** `src/training/losses.py`

```python
"""Loss functions for segmentation."""

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    
    Loss = 1 - Dice coefficient
    """
    
    def __init__(self, smooth: float = 1e-6):
        """Initialize Dice loss."""
        pass
    
    def forward(self, pred, target):
        """Calculate loss."""
        pass

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss for imbalanced segmentation.
    
    Combines Tversky index with focal mechanism
    to handle class imbalance.
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, 
                 gamma: float = 0.75, smooth: float = 1e-6):
        """
        Initialize Focal Tversky loss.
        
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            gamma: Focusing parameter
            smooth: Smoothing constant
        """
        pass
    
    def forward(self, pred, target):
        """Calculate loss."""
        pass

class BCELoss(nn.Module):
    """Binary cross-entropy loss."""
    
    def __init__(self):
        """Initialize BCE loss."""
        pass
    
    def forward(self, pred, target):
        """Calculate loss."""
        pass

def create_loss(loss_config: dict) -> nn.Module:
    """
    Create loss function from configuration.
    
    Args:
        loss_config: Loss configuration dictionary
    
    Returns:
        Loss function instance
    """
    pass
```

### 6.3 Metrics

**File:** `src/training/metrics.py`

```python
"""Evaluation metrics for segmentation."""

import torch
from typing import Dict, Callable

def dice_coefficient(pred, target, smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient.
    
    Args:
        pred: Predicted segmentation
        target: Ground truth
        smooth: Smoothing constant
    
    Returns:
        Dice score
    """
    pass

def iou_score(pred, target, smooth: float = 1e-6) -> float:
    """
    Calculate IoU (Jaccard index).
    
    Args:
        pred: Predicted segmentation
        target: Ground truth
        smooth: Smoothing constant
    
    Returns:
        IoU score
    """
    pass

def precision(pred, target, smooth: float = 1e-6) -> float:
    """Calculate precision."""
    pass

def recall(pred, target, smooth: float = 1e-6) -> float:
    """Calculate recall (sensitivity)."""
    pass

def create_metrics(metric_names: list) -> Dict[str, Callable]:
    """
    Create metric functions from names.
    
    Args:
        metric_names: List of metric names
    
    Returns:
        Dictionary mapping names to metric functions
    """
    pass
```

---

## 7. Testing & Inference

### 7.1 Testing Script

**File:** `scripts/test.py`

```python
"""
Testing script for evaluating trained models.

Usage:
    python scripts/test.py --config path/to/config.yaml
"""

import argparse

def test(config_path: str, checkpoint_path: str = None):
    """
    Run testing on test set.
    
    Args:
        config_path: Path to experiment config
        checkpoint_path: Path to model checkpoint (default: best.pth)
    
    Workflow:
    1. Load configuration
    2. Create test dataloader
    3. Load model and checkpoint
    4. Run inference on test set
    5. Calculate metrics
    6. Save predictions (if configured)
    7. Save metrics to JSON
    """
    pass

def main():
    """Command-line interface."""
    pass

if __name__ == '__main__':
    main()
```

### 7.2 Inference Module

**File:** `src/inference/predictor.py`

```python
"""Inference engine for model predictions."""

import torch
import torch.nn as nn
from typing import Dict
import numpy as np

class Predictor:
    """
    Inference engine for trained models.
    
    Responsibilities:
    - Load trained model
    - Run inference on single images or batches
    - Post-process predictions
    - Handle different output formats
    """
    
    def __init__(self, model: nn.Module, device, config: dict):
        """
        Initialize predictor.
        
        Args:
            model: Trained model
            device: Device for inference
            config: Model configuration
        """
        pass
    
    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict on a single image.
        
        Args:
            image: Input image (H, W, C) as numpy array
        
        Returns:
            Dictionary with:
                - "mask": Binary prediction (H, W)
                - "probability": Probability map (H, W)
        """
        pass
    
    def predict_batch(self, images):
        """Predict on batch of images."""
        pass
```

---

## 8. Utilities

### 8.1 Logger

**File:** `src/utils/logger.py`

```python
"""TensorBoard logging utilities."""

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict

class TensorBoardLogger:
    """
    TensorBoard logging manager.
    
    Responsibilities:
    - Log scalar metrics (loss, accuracy, etc.)
    - Log images (inputs, predictions, ground truth)
    - Log learning rate
    - Organize logs by experiment
    """
    
    def __init__(self, config: dict):
        """
        Initialize logger.
        
        Args:
            config: Experiment configuration
        """
        pass
    
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float],
                  val_metrics: Dict[str, float]):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        pass
    
    def log_images(self, epoch: int, images, masks, predictions):
        """
        Log sample images.
        
        Args:
            epoch: Epoch number
            images: Input images
            masks: Ground truth masks
            predictions: Model predictions
        """
        pass
    
    def close(self):
        """Close logger."""
        pass
```

### 8.2 Helper Functions

**File:** `src/utils/helpers.py`

```python
"""Miscellaneous utility functions."""

import torch
import random
import numpy as np

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    pass

def count_parameters(model) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    pass

def save_predictions(predictions, output_dir: str):
    """
    Save predictions as images.
    
    Args:
        predictions: Model predictions
        output_dir: Output directory
    """
    pass
```

---

## 9. Command-Line Interface

### 9.1 Training Script

**File:** `scripts/train.py`

```python
"""
Main training script.

Usage:
    python scripts/train.py --config configs/experiments/exp001.yaml
"""

import argparse
import sys
from pathlib import Path

def main(config_path: str):
    """
    Main training function.
    
    Workflow:
    1. Load configuration
    2. Set random seed
    3. Create output directory
    4. Save configuration copy
    5. Create dataloaders
    6. Create model
    7. Create trainer
    8. Run training
    9. Report results
    
    Args:
        config_path: Path to experiment configuration
    """
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train vessel segmentation model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment config file')
    args = parser.parse_args()
    
    main(args.config)
```

### 9.2 Shell Launcher

**File:** `train.sh`

```bash
#!/bin/bash
# Simple training launcher script

# Usage: ./train.sh <experiment_name>

set -e

CONFIG_DIR="configs/experiments"

if [ $# -eq 0 ]; then
    echo "Usage: ./train.sh <experiment_name>"
    echo ""
    echo "Available experiments:"
    ls -1 "$CONFIG_DIR"/*.yaml | xargs -n 1 basename | sed 's/.yaml//'
    exit 1
fi

EXPERIMENT=$1
CONFIG_FILE="$CONFIG_DIR/${EXPERIMENT}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Training: $EXPERIMENT"
echo "Config: $CONFIG_FILE"
echo "=========================================="

python scripts/train.py --config "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
```

---

## 10. Implementation Checklist

### Phase 1: Foundation
- [ ] Set up project structure
- [ ] Create configuration system
  - [ ] Dataset YAML loading
  - [ ] Experiment YAML loading
  - [ ] Config validation
- [ ] Implement basic dataset class
  - [ ] Image and mask loading
  - [ ] Basic preprocessing
- [ ] Implement model registry
- [ ] Implement basic UNet
  - [ ] Encoder path
  - [ ] Bottleneck
  - [ ] Decoder path with skip connections
  - [ ] Output head

### Phase 2: Training Pipeline
- [ ] Implement loss functions
  - [ ] Dice loss
  - [ ] Focal Tversky loss
- [ ] Implement metrics
  - [ ] Dice coefficient
  - [ ] IoU
  - [ ] Precision
  - [ ] Recall
- [ ] Implement Trainer class
  - [ ] Training loop
  - [ ] Validation loop
  - [ ] Mixed precision support
  - [ ] Gradient clipping
- [ ] Implement optimizer creation
- [ ] Implement scheduler creation
- [ ] Implement checkpointing
- [ ] Implement early stopping

### Phase 3: Testing & Utilities
- [ ] Implement testing script
- [ ] Implement predictor class
- [ ] Implement TensorBoard logger
  - [ ] Scalar logging
  - [ ] Image logging
- [ ] Implement helper functions
- [ ] Create shell launcher script

### Phase 4: Documentation & Polish
- [ ] Write README
- [ ] Add docstrings
- [ ] Create example configs
- [ ] Test full workflow
- [ ] Write requirements.txt

### Phase 5: Extensions (Future)
- [ ] Implement data augmentation
  - [ ] Geometric transforms
  - [ ] Intensity transforms
  - [ ] Noise augmentation
- [ ] Implement attention mechanisms
  - [ ] Attention gates
  - [ ] Spatial attention
- [ ] Implement transformer blocks
  - [ ] Transformer layers
  - [ ] Transformer bottleneck
- [ ] Add more architectures
  - [ ] Attention UNet
  - [ ] TransUNet
- [ ] Add more losses and metrics

---

## Summary

This specification defines a clean, modular pipeline for vessel segmentation with the following characteristics:

**Configuration:**
- Two-level: Dataset (static) + Experiment (dynamic)
- Self-contained experiment configs
- Easy to understand and modify

**Architecture:**
- Model registry for easy extension
- Composable building blocks
- Support for transformers and attention

**Training:**
- Mixed precision support
- Flexible optimizer and scheduler
- Multiple loss functions
- Comprehensive metrics

**Workflow:**
```bash
# Create experiment config
nano configs/experiments/my_experiment.yaml

# Train
./train.sh my_experiment

# Test
python scripts/test.py --config outputs/experiments/my_experiment/config.yaml

# View logs
tensorboard --logdir outputs/experiments/my_experiment/logs
```

This design prioritizes **clarity** and **extensibility** while keeping the initial implementation **simple** and **focused**. 