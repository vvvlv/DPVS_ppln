# Vessel Segmentation Pipeline

A clean, modular deep learning pipeline for training UNet models on fundus vessel segmentation tasks (FIVES dataset).

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your data (see Data Setup below)

# 3. Train a model
./train.sh exp001_basic_unet

# 4. Test the model
./test.sh exp001_basic_unet
```

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Data Setup](#-data-setup)
- [Training](#-training)
- [Testing & Inference](#-testing--inference)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Output Structure](#-output-structure)
- [Extending the Pipeline](#-extending-the-pipeline)

---

## ✨ Features

### Core Functionality
- ✅ **Modular Configuration System**: YAML-based dataset + experiment configs
- ✅ **UNet Architecture**: Flexible 5-level encoder-decoder with skip connections
- ✅ **Training Loop**: Complete with validation, metrics tracking, and progress bars
- ✅ **Early Stopping**: Stops training when validation metrics stop improving (with patience)
- ✅ **Metrics History**: Saves all epoch metrics to YAML for easy analysis
- ✅ **Checkpointing**: Saves best and last model checkpoints
- ✅ **Testing & Inference**: Load checkpoints, run predictions, save masks and metrics

### Loss Functions & Metrics
- **Loss**: Dice Loss (smooth, differentiable)
- **Metrics**: Dice Coefficient, IoU (Intersection over Union)
- **Per-Image Metrics**: Individual metrics for each test image

### Data Handling
- **Dataset**: Automatic loading of images and masks
- **Preprocessing**: Normalization, padding to multiples of 32
- **Image Format**: Supports PNG images

---

## 🔧 Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- ~2.7 GB disk space for dependencies

### Quick Install

```bash
cd /home/vlv/Documents/master/deepLearning/project/codebase
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

For detailed installation instructions and troubleshooting, see [INSTALL.md](INSTALL.md).

---

## 📁 Data Setup

Place your FIVES dataset in the following structure:

```
codebase/data/FIVES512/
├── train/
│   ├── image/          # Training images (*.png)
│   └── label/          # Training masks (*.png)
├── val/
│   ├── image/          # Validation images
│   └── label/          # Validation masks
└── test/
    ├── image/          # Test images
    └── label/          # Test masks
```

### Custom Data Path

To use a different path, edit `configs/datasets/fives_512.yaml`:

```yaml
paths:
  root: "/your/custom/path/to/FIVES512"
  train: "/your/custom/path/to/FIVES512/train"
  val: "/your/custom/path/to/FIVES512/val"
  test: "/your/custom/path/to/FIVES512/test"
```

---

## 🎓 Training

### Using Shell Script (Recommended)

```bash
# Make script executable (first time only)
chmod +x train.sh

# Train with experiment config
./train.sh exp001_basic_unet
```

### Using Python Directly

```bash
python scripts/train.py --config configs/experiments/exp001_basic_unet.yaml
```

### What Happens During Training

1. **Initialization**: Loads config, creates model, sets random seed
2. **Training Loop**: 
   - Trains on training set with progress bar
   - Validates after each epoch
   - Prints metrics (loss, dice, IoU)
   - Saves metrics history to YAML after each epoch
3. **Checkpointing**: 
   - Saves best model when validation metric improves
   - Saves last checkpoint every epoch
4. **Early Stopping**: 
   - Monitors validation metric (e.g., val_dice)
   - Stops training if no improvement for N epochs (patience)
   - Displays countdown during no-improvement periods

### Training Output

```
Starting training for 20 epochs...

Epoch 1 [Train]: 100%|████████████| 150/150 [01:23<00:00]
Epoch 1 [Val]:   100%|████████████| 30/30 [00:12<00:00]

Epoch 1/20
  Train - Loss: 0.3456, dice: 0.6544, iou: 0.5123
  Val   - Loss: 0.3789, dice: 0.6211, iou: 0.4890
  → New best val_dice: 0.6211

Epoch 2/20
  Train - Loss: 0.2987, dice: 0.7013, iou: 0.5567
  Val   - Loss: 0.3234, dice: 0.6766, iou: 0.5234
  → New best val_dice: 0.6766
...
```

All epoch metrics are automatically saved to `outputs/experiments/<exp_name>/metrics_history.yaml`.

---

## 🧪 Testing & Inference

### Using Shell Script (Recommended)

```bash
# Test with best checkpoint
./test.sh exp001_basic_unet

# Test with last checkpoint
./test.sh exp001_basic_unet last
```

### Using Python Directly

```bash
python scripts/test.py --config configs/experiments/exp001_basic_unet.yaml
```

### What the Test Script Does

1. **Loads** the trained model checkpoint (best.pth or last.pth)
2. **Runs inference** on all test images with progress bar
3. **Calculates metrics** for each image (Dice, IoU)
4. **Saves predicted masks** to `outputs/tests/<exp_name>/predictions/`
5. **Saves metrics** to YAML files:
   - `test_metrics.yaml` - Average metrics across all test images
   - `per_image_metrics.yaml` - Individual metrics for each image

### Test Output Structure

```
outputs/tests/exp001_basic_unet/
├── predictions/              # Predicted segmentation masks
│   ├── 1_A.png
│   ├── 2_A.png
│   └── ...
├── test_metrics.yaml        # Summary: average metrics
└── per_image_metrics.yaml   # Detailed: per-image metrics
```

### Example Metrics Files

**test_metrics.yaml:**
```yaml
experiment: exp001_basic_unet
checkpoint: outputs/experiments/exp001_basic_unet/checkpoints/best.pth
num_test_images: 200
average_metrics:
  dice: 0.7834
  iou: 0.6912
```

**per_image_metrics.yaml:**
```yaml
- image: 1_A.png
  dice: 0.7912
  iou: 0.7034
- image: 2_A.png
  dice: 0.8123
  iou: 0.7245
...
```

---

## ⚙️ Configuration

The pipeline uses a **two-level configuration system**:

### 1. Dataset Configuration (Static)

**File**: `configs/datasets/fives_512.yaml`

Defines dataset properties that rarely change:
- Data paths
- Image dimensions
- Normalization statistics

```yaml
name: "FIVES512"
paths:
  root: "data/FIVES512"
  train: "data/FIVES512/train"
  val: "data/FIVES512/val"
  test: "data/FIVES512/test"
image_size: [512, 512]
num_channels: 3
num_classes: 1
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]
```

### 2. Experiment Configuration (Dynamic)

**File**: `configs/experiments/exp001_basic_unet.yaml`

Contains all training parameters for an experiment:

```yaml
name: "exp001_basic_unet"
dataset: "configs/datasets/fives_512.yaml"

# Data loading
data:
  batch_size: 4
  num_workers: 2
  pin_memory: true

# Model architecture
model:
  type: "UNet"
  in_channels: 3
  out_channels: 1
  depths: [32, 64, 128, 256, 512]
  final_activation: "sigmoid"

# Training settings
training:
  epochs: 20
  optimizer:
    type: "adam"
    learning_rate: 0.0001
    weight_decay: 0.0001
  scheduler:
    type: "cosine"
    min_lr: 0.000001
  loss:
    type: "dice"
    smooth: 0.000001
  metrics:
    - "dice"
    - "iou"
  early_stopping:
    enabled: true
    patience: 5
    monitor: "val_dice"
    mode: "max"

# Output
output:
  dir: "outputs/experiments/exp001_basic_unet"
  save_predictions: true

seed: 42
device: "cuda"
```

### Common Modifications

**Change batch size** (for GPU memory):
```yaml
data:
  batch_size: 8  # Increase if you have more GPU memory
```

**Adjust learning rate**:
```yaml
training:
  optimizer:
    learning_rate: 0.001  # Larger for faster convergence
```

**Change model size**:
```yaml
model:
  depths: [16, 32, 64, 128, 256]  # Smaller model for less memory
```

**Adjust early stopping**:
```yaml
training:
  early_stopping:
    enabled: true
    patience: 10  # Wait 10 epochs before stopping
```

### Create New Experiment

```bash
# Copy existing config
cp configs/experiments/exp001_basic_unet.yaml configs/experiments/exp002_my_test.yaml

# Edit the new config
nano configs/experiments/exp002_my_test.yaml

# Train with new config
./train.sh exp002_my_test
```

---

## 📂 Project Structure

```
codebase/
├── configs/
│   ├── datasets/              # Dataset configurations
│   │   └── fives_512.yaml
│   └── experiments/           # Experiment configurations
│       └── exp001_basic_unet.yaml
│
├── src/                       # Source code
│   ├── data/
│   │   ├── dataset.py        # Dataset class
│   │   └── __init__.py       # DataLoader factory
│   ├── models/
│   │   ├── registry.py       # Model registration system
│   │   ├── architectures/
│   │   │   └── unet.py       # UNet implementation
│   │   └── blocks/
│   │       └── conv_blocks.py # Building blocks
│   ├── training/
│   │   ├── trainer.py        # Training loop
│   │   ├── losses.py         # Loss functions
│   │   └── metrics.py        # Metrics
│   └── utils/
│       ├── config.py         # Config loading
│       └── helpers.py        # Utilities
│
├── scripts/
│   ├── train.py              # Training script
│   └── test.py               # Testing script
│
├── data/                      # Data directory (gitignored)
│   └── FIVES512/
│
├── outputs/                   # Training outputs (gitignored)
│   ├── experiments/          # Training results
│   └── tests/                # Test results
│
├── train.sh                   # Training launcher
├── test.sh                    # Testing launcher
├── requirements.txt           # Dependencies
├── INSTALL.md                # Installation guide
└── README.md                 # This file
```

---

## 📊 Output Structure

### Training Outputs

```
outputs/experiments/exp001_basic_unet/
├── config.yaml              # Copy of experiment config
├── checkpoints/
│   ├── best.pth            # Best model (highest val_dice)
│   └── last.pth            # Latest checkpoint
└── metrics_history.yaml    # All epoch metrics
```

**metrics_history.yaml** format:
```yaml
- epoch: 1
  train:
    loss: 0.3456
    dice: 0.6544
    iou: 0.5123
  val:
    val_loss: 0.3789
    val_dice: 0.6211
    val_iou: 0.4890
- epoch: 2
  train:
    loss: 0.2987
    dice: 0.7013
    iou: 0.5567
  val:
    val_loss: 0.3234
    val_dice: 0.6766
    val_iou: 0.5234
...
```

### Test Outputs

```
outputs/tests/exp001_basic_unet/
├── predictions/             # Predicted masks (PNG images)
│   ├── 1_A.png
│   ├── 2_A.png
│   └── ...
├── test_metrics.yaml       # Average metrics
└── per_image_metrics.yaml  # Per-image metrics
```

---

## 🔬 Extending the Pipeline

### Add a New Loss Function

**File**: `src/training/losses.py`

```python
class MyCustomLoss(nn.Module):
    def __init__(self, param=1.0):
        super().__init__()
        self.param = param
    
    def forward(self, pred, target):
        # Your loss calculation
        return loss

# Add to create_loss function
def create_loss(loss_config):
    loss_type = loss_config['type']
    if loss_type == 'my_custom':
        return MyCustomLoss(param=loss_config.get('param', 1.0))
    # ... existing losses
```

Then use in config:
```yaml
training:
  loss:
    type: "my_custom"
    param: 2.0
```

### Add a New Metric

**File**: `src/training/metrics.py`

```python
def my_custom_metric(pred, target, threshold=0.5):
    # Your metric calculation
    return metric_value.item()

# Add to METRICS dictionary
METRICS = {
    'dice': dice_coefficient,
    'iou': iou_score,
    'my_metric': my_custom_metric
}
```

Then use in config:
```yaml
training:
  metrics:
    - "dice"
    - "iou"
    - "my_metric"
```

### Add a New Model

**File**: `src/models/architectures/my_model.py`

```python
from ..registry import register_model
import torch.nn as nn

@register_model('MyModel')
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Your architecture
        
    def forward(self, x):
        # Forward pass
        return output
```

Import in `src/models/__init__.py`:
```python
from .architectures.unet import UNet
from .architectures.my_model import MyModel  # Add this
```

Use in config:
```yaml
model:
  type: "MyModel"
  # ... your model parameters
```

---

## 🐛 Troubleshooting

### "No images found in..."
- Check data path in `configs/datasets/fives_512.yaml`
- Ensure images are `.png` format
- Verify folder structure matches expected layout

### CUDA Out of Memory
- Reduce `batch_size` in experiment config
- Reduce model size: `depths: [16, 32, 64, 128, 256]`
- Close other GPU applications

### Import Errors
- Run from codebase root directory
- Verify all `__init__.py` files exist
- Check dependencies are installed: `pip list`

### Training Not Improving
- Check learning rate (try 0.001 or 0.0001)
- Verify data is normalized correctly
- Check loss function is appropriate for your task
- Increase number of epochs

### Early Stopping Too Soon
- Increase patience in config:
  ```yaml
  early_stopping:
    patience: 10  # or higher
  ```
- Check if validation set is representative

---

## 📈 Expected Performance

With default configuration on FIVES512:

| Metric | Value |
|--------|-------|
| **Training Time** | ~1-2 min/epoch (NVIDIA T4) |
| **GPU Memory** | ~4-6 GB |
| **Model Parameters** | ~7.8M |
| **Expected Dice** | 0.70-0.80+ after 10-20 epochs |
| **Expected IoU** | 0.60-0.70+ after 10-20 epochs |

---

## 🎯 Key Design Decisions

1. **Configuration-Driven**: All parameters in YAML files, no hardcoded values
2. **Modular**: Each component is independent and replaceable
3. **Reproducible**: Seed management, config saving, deterministic operations
4. **Extensible**: Easy to add new models, losses, metrics
5. **User-Friendly**: Shell scripts for common operations, clear error messages

---

## 📝 Citation

If you use this pipeline, please cite the FIVES dataset:

```bibtex
@article{fives2022,
  title={FIVES: A Fundus Image Dataset for Vessel Segmentation},
  journal={Scientific Data},
  year={2022}
}
```

---

## 📧 Support

For issues, questions, or contributions:
1. Check this README and INSTALL.md first
2. Review your configuration files
3. Check error messages carefully
4. Verify data paths and formats

---

**Happy Training! 🚀**