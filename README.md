# Vessel Segmentation Pipeline

A clean, modular deep learning pipeline for training UNet models on fundus vessel segmentation tasks (FIVES dataset).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your data (see Data Setup below)

# 3. Train a model
./train.sh exp001_basic_unet  # UNet baseline
# OR
./train.sh exp002_roinet      # RoiNet with residuals
# OR
./train.sh exp003_utrans      # UTrans (UNet + Transformer)
# OR
./train.sh exp004_transroinet # TransRoiNet (RoiNet + Transformer)

# 4. Test the model
./test.sh exp001_basic_unet
```

## Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Data Setup](#-data-setup)
- [Training](#-training)
- [Testing & Inference](#-testing--inference)
- [TensorBoard Visualization](#-tensorboard-visualization)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Output Structure](#-output-structure)
- [Memory Profiling & Debugging](#-memory-profiling--debugging)
- [Extending the Pipeline](#-extending-the-pipeline)

---

## Features

### Core Functionality
- Modular Configuration System: YAML-based dataset + experiment configs
- Multiple Architectures: 
  - UNet: Classic encoder-decoder
  - RoiNet: Residual blocks with deepened bottleneck
  - UTrans: UNet + Transformer for global context
  - TransRoiNet: RoiNet + Transformer (best of both worlds)
- Reusable Transformer Blocks: Modular attention components for building hybrid models
- Training Loop: Complete with validation, metrics tracking, and progress bars
- TensorBoard Integration: Real-time visualization of training metrics, learning curves, and predictions
- Memory Profiling: Comprehensive VRAM usage analysis for debugging and optimization
- Early Stopping: Stops training when validation metrics stop improving (with patience)
- Metrics History: Saves all epoch metrics to YAML for easy analysis
- Checkpointing: Saves best and last model checkpoints
- Testing & Inference: Load checkpoints, run predictions, save masks and metrics

### Loss Functions & Metrics
- Loss: Dice Loss (smooth, differentiable)
- Metrics: Dice Coefficient, IoU (Intersection over Union), AUC (Area Under ROC Curve)
- Per-Image Metrics: Individual metrics for each test image
- Advanced Logging: Layer activation monitoring and histograms

### Data Handling
- Dataset: Automatic loading of images and masks
- Preprocessing: Normalization, padding to multiples of 32
- Image Format: Supports PNG images

---

## Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- ~3.5 GB disk space for dependencies

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

## Data Setup

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

## Training

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

1. Initialization: Loads config, creates model, sets random seed
2. Training Loop: 
   - Trains on training set with progress bar
   - Validates after each epoch
   - Prints metrics (loss, dice, IoU)
   - Saves metrics history to YAML after each epoch
3. Checkpointing: 
   - Saves best model when validation metric improves
   - Saves last checkpoint every epoch
4. Early Stopping: 
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

## TensorBoard Visualization

### Overview

TensorBoard integration is fully supported for real-time training visualization and experiment tracking.

### Enabling TensorBoard

TensorBoard is controlled via the `logging` section in your experiment config:

```yaml
logging:
  tensorboard: true           # Enable/disable TensorBoard logging
  log_images: false           # Enable/disable image logging (optional)
  image_log_frequency: 5      # Log images every N epochs (default: 5, set to 1 for every epoch)
```

**Note**: All existing experiment configs already have `tensorboard: true` by default.

### Starting TensorBoard

**During Training**:
```bash
# In a separate terminal, run:
tensorboard --logdir outputs/experiments/<exp_name>/tensorboard

# For multiple experiments:
tensorboard --logdir outputs/experiments

# Then open in browser:
# http://localhost:6006
```

**After Training**:
```bash
# View logs for a specific experiment
tensorboard --logdir outputs/experiments/exp001_basic_unet/tensorboard

# Compare multiple experiments
tensorboard --logdir outputs/experiments
```

### What's Logged

#### 1. **Scalars** (Automatically Logged Every Epoch)

**Training Metrics**:
- `train/loss` - Training loss
- `train/dice` - Training Dice coefficient
- `train/iou` - Training IoU score

**Validation Metrics**:
- `val/loss` - Validation loss
- `val/dice` - Validation Dice coefficient
- `val/iou` - Validation IoU score

**Learning Rate**:
- `learning_rate` - Current learning rate (tracks scheduler)

**Comparison Plots**:
- `comparison/dice` - Train vs Val Dice on same plot
- `comparison/iou` - Train vs Val IoU on same plot

#### 2. **Images** (Optional, Enabled with `log_images: true`)

When `log_images: true` in config:
- **Frequency**: Configurable via `image_log_frequency` (default: 5 epochs), or when best model is saved
- **Content**: Side-by-side comparison of:
  - Input image (denormalized)
  - Ground truth mask
  - Predicted mask
- **Location**: `val/predictions` tab
- **Samples**: Up to 4 validation samples per log

**Example**: To log images every epoch:
```yaml
logging:
  tensorboard: true
  log_images: true
  image_log_frequency: 1  # Log every epoch
```

#### 3. **Layer Activations** (Optional, Enabled with `log_activations: true`)

Monitor neural network layer activations:
- **Frequency**: Configurable via `activation_log_frequency` (default: 5 epochs)
- **Content**: For each monitored layer:
  - Histogram of activation values
  - Statistics (mean, std, min, max)
- **Location**: `Histograms` tab (distributions), `Scalars` tab (statistics)
- **Layer Selection**: 
  - `"auto"`: Model-specific defaults (recommended)
  - Custom list: Specify exact layers
  - `null`: Monitor all layers (not recommended)

**Example**: To log activations every epoch:
```yaml
logging:
  tensorboard: true
  log_activations: true
  activation_log_frequency: 1
  activation_layers: "auto"      # Or specify: ["encoder1", "bottleneck"]
```

#### 4. **Model Graph**

The model architecture graph is automatically logged at training start:
- Shows layer connections and data flow
- Useful for debugging model structure
- View in the "Graphs" tab

#### 4. **Hyperparameters**

At training completion, logs hyperparameters and final metrics:
- Model type, batch size, learning rate, etc.
- Final validation metrics
- Best metric achieved
- Enables comparison across experiments

### TensorBoard Features

**Scalars Tab**:
- Smooth curves (adjust smoothing slider)
- Compare runs side-by-side
- Toggle specific runs on/off
- Download data as CSV/JSON

**Images Tab**:
- View prediction quality over time
- Identify overfitting visually
- Track model convergence

**Graphs Tab**:
- Visualize model architecture
- Verify layer connections

**HParams Tab**:
- Compare hyperparameters across experiments
- Identify best configurations
- Parallel coordinates plot

### Example Workflow

1. **Start training with TensorBoard enabled**:
   ```bash
   ./train.sh exp001_basic_unet
   ```

2. **In a separate terminal, start TensorBoard**:
   ```bash
   tensorboard --logdir outputs/experiments/exp001_basic_unet/tensorboard
   ```

3. **Open browser**:
   - Navigate to `http://localhost:6006`
   - Watch metrics update in real-time as training progresses

4. **Compare multiple experiments**:
   ```bash
   # Run multiple experiments
   ./train.sh exp001_basic_unet
   ./train.sh exp002_roinet
   ./train.sh exp003_utrans
   
   # View all together
   tensorboard --logdir outputs/experiments
   ```

### Test Results in TensorBoard

Test metrics are also logged to TensorBoard when running tests:

```bash
./test.sh exp001_basic_unet
```

Test logs are saved to: `outputs/experiments/<exp_name>/tensorboard/test/`

### Disabling TensorBoard

To disable TensorBoard for an experiment:

```yaml
logging:
  tensorboard: false    # Disable TensorBoard
  log_images: false
```

Training will proceed normally with only console output and YAML metrics files.

### Tips

1. **Remote Server**: If training on a remote server, use SSH port forwarding:
   ```bash
   ssh -L 6006:localhost:6006 user@remote-server
   tensorboard --logdir /path/to/experiments
   ```

2. **Custom Port**: Use a different port if 6006 is occupied:
   ```bash
   tensorboard --logdir outputs/experiments --port 6007
   ```

3. **Multiple Instances**: Run multiple TensorBoard instances for different experiment groups:
   ```bash
   tensorboard --logdir outputs/experiments/baseline_models --port 6006
   tensorboard --logdir outputs/experiments/transformer_models --port 6007
   ```

4. **Refresh**: TensorBoard auto-refreshes every 30 seconds. Click the refresh button for immediate updates.

---

## Testing & Inference

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

1. Loads the trained model checkpoint (best.pth or last.pth)
2. Runs inference on all test images with progress bar
3. Calculates metrics for each image (Dice, IoU)
4. Saves predicted masks to `outputs/tests/<exp_name>/predictions/`
5. Saves metrics to YAML files:
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

## Configuration

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
  
  augmentation:
    enabled: false
  
  preprocessing:
    normalize: true
    pad_to_multiple: 32

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

## Project Structure

```
codebase/
├── configs/
│   ├── datasets/              # Dataset configurations
│   │   └── fives_512.yaml
│   └── experiments/           # Experiment configurations
│       ├── exp001_basic_unet.yaml
│       ├── exp002_roinet.yaml
│       ├── exp002_roinet_batch_size.yaml
│       ├── exp003_utrans.yaml
│       └── exp004_transroinet.yaml
│
├── src/                       # Source code
│   ├── data/
│   │   ├── dataset.py        # Dataset class
│   │   └── __init__.py       # DataLoader factory
│   ├── models/
│   │   ├── registry.py       # Model registration system
│   │   ├── architectures/
│   │   │   ├── unet.py       # UNet implementation
│   │   │   ├── roinet.py     # RoiNet implementation
│   │   │   ├── utrans.py     # UTrans implementation (UNet + Transformer)
│   │   │   └── transroinet.py # TransRoiNet implementation (RoiNet + Transformer)
│   │   └── blocks/
│   │       ├── conv_blocks.py      # CNN blocks (DoubleConv, ResidualBlock)
│   │       └── transformer_blocks.py # Transformer blocks (Self-Attention, FFN, etc.)
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

## Output Structure

### Training Outputs

```
outputs/experiments/exp001_basic_unet/
├── config.yaml              # Copy of experiment config
├── checkpoints/
│   ├── best.pth            # Best model (highest val_dice)
│   └── last.pth            # Latest checkpoint
├── metrics_history.yaml    # All epoch metrics
└── tensorboard/            # TensorBoard logs (if enabled)
    ├── events.out.tfevents.*
    └── test/               # Test results (if test.py was run)
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

## Available Models

### UNet
Classic encoder-decoder architecture with skip connections:
- Structure: 5-level pyramid with symmetric encoder-decoder
- Building Block: DoubleConv (Conv + BN + ReLU × 2)
- Default Depths: [32, 64, 128, 256, 512]
- Parameters: ~7.8M
- Best for: Standard segmentation tasks, baseline experiments

### RoiNet
Advanced architecture with residual connections and deepened bottleneck:
- Structure: 3-level encoder-decoder with residual blocks
- Building Block: ResidualBlock with configurable kernel size
- Default Depths: [32, 64, 128, 128, 64, 32]
- Kernel Size: 9 (configurable, larger receptive field)
- Bottleneck: Deepened with 2 additional residual blocks
- Parameters: Varies with configuration
- Best for: Complex features, better gradient flow via residuals

### UTrans 
Hybrid CNN-Transformer architecture combining local and global features:
- Structure: UNet encoder-decoder with Transformer bottleneck
- Encoder/Decoder: 4-level CNN with DoubleConv blocks
- Bottleneck: Multi-head self-attention transformer (configurable depth)
- Default Depths: [64, 128, 256, 512, 1024]
- Transformer Config: 4 blocks, 8 heads, MLP ratio 4.0
- Parameters: ~100M (actual: 100,300,993)
- Key Feature: Captures long-range dependencies while preserving local details
- Best for: Tasks requiring global context (large vessels, complex structures)

### TransRoiNet 
Advanced hybrid combining RoiNet's residuals with Transformer attention:
- Structure: RoiNet encoder-decoder with Transformer-enhanced bottleneck
- Encoder/Decoder: 3-level with ResidualBlocks (k_size=9)
- Bottleneck: Residual → Transformer → Residual → Merge
- Default Depths: [32, 64, 128, 128, 64, 32]
- Transformer Config: 2 blocks, 8 heads (lighter than UTrans)
- Parameters: ~35-40M (depends on configuration)
- Key Features: 
  - Residual connections for stable training
  - Transformer captures vessel connectivity patterns
  - Large receptive field from k_size=9
- Best for: Complex segmentation requiring both local precision and global context

### Using Different Models

Just change the `model.type` in your experiment config:

```yaml
# For UNet
model:
  type: "UNet"
  in_channels: 3
  out_channels: 1
  depths: [32, 64, 128, 256, 512]
  final_activation: "sigmoid"

# For RoiNet
model:
  type: "RoiNet"
  in_channels: 3
  out_channels: 1
  depths: [32, 64, 128, 128, 64, 32]
  kernel_size: 9
  final_activation: "sigmoid"

# For UTrans (UNet + Transformer)
model:
  type: "UTrans"
  in_channels: 3
  out_channels: 1
  depths: [64, 128, 256, 512, 1024]
  transformer_depth: 4        # Number of transformer blocks
  transformer_heads: 8        # Attention heads
  transformer_mlp_ratio: 4.0  # FFN expansion ratio
  transformer_dropout: 0.1    # Dropout probability
  final_activation: "sigmoid"

# For TransRoiNet (RoiNet + Transformer)
model:
  type: "TransRoiNet"
  in_channels: 3
  out_channels: 1
  depths: [32, 64, 128, 128, 64, 32]
  kernel_size: 9              # Large kernel for residual blocks
  transformer_depth: 2        # Number of transformer blocks (lighter)
  transformer_heads: 8        # Attention heads
  transformer_mlp_ratio: 4.0  # FFN expansion ratio
  transformer_dropout: 0.1    # Dropout probability
  final_activation: "sigmoid"
```

---

## Extending the Pipeline

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

## Memory Profiling & Debugging

The pipeline includes comprehensive VRAM profiling to help debug memory issues and optimize resource usage.

### Quick Enable

Add to your experiment config:

```yaml
debug:
  profile_memory: true              # Enable memory profiling
  detailed_memory: true              # Show per-layer breakdown
  estimate_activations: true         # Estimate activation memory
  profile_training_step: false       # Profile actual training step (CUDA only)
```

### What You Get

Before training starts, you'll see:

```
📊 GPU Memory (Device: cuda:0):
  Total VRAM:      23.65 GB
  Currently Used:  245.67 MB
  Available:       23.41 GB

🧠 Model Memory Breakdown:
  Parameters:      93.52 MB
  Buffers:         0.12 MB
  Total Model:     93.64 MB

⚙️  Training Memory Estimates:
  Gradients:       93.52 MB
  Optimizer (ADAM): 187.04 MB
  Activations:     1.23 GB

💾 Total Estimated Training Memory: 1.59 GB
   (~6.7% of available VRAM)

📋 Per-Layer Memory Breakdown (Top 15):
  Layer Name                               Parameters      Memory      
  ---------------------------------------- --------------- ------------
  encoder4                                 8,388,608       32.00 MB
  encoder3                                 2,097,152       8.00 MB
  ...
```

### Example Usage

Enable in any experiment config (e.g., `exp001_basic_unet.yaml`):

```yaml
debug:
  profile_memory: true              # Enable profiling
  detailed_memory: true              # Show per-layer breakdown
  estimate_activations: true         # Estimate activation memory
  profile_training_step: false       # Optional: profile training step (CUDA only)
```

Then run normally:

```bash
python scripts/train.py --config configs/experiments/exp001_basic_unet.yaml
```

### Use Cases

1. **Out of Memory Errors**: See exactly what's consuming VRAM
2. **Optimize Batch Size**: Calculate maximum safe batch size
3. **Model Comparison**: Compare memory usage across architectures
4. **Layer Analysis**: Identify memory-heavy layers

### Tips

- Set `profile_memory: true` when debugging memory issues
- Use `profile_training_step: true` (CUDA only) to see memory at each training stage
- Set to `false` once debugging is complete to skip overhead

---

## Troubleshooting

### "No images found in..."
- Check data path in `configs/datasets/fives_512.yaml`
- Ensure images are `.png` format
- Verify folder structure matches expected layout

### CUDA Out of Memory
- **First**: Enable memory profiling to see what's using VRAM:
  ```yaml
  debug:
    profile_memory: true
  ```
- Reduce `batch_size` in experiment config
- Reduce model size: `depths: [16, 32, 64, 128, 256]`
- Close other GPU applications
- Consider using gradient accumulation or mixed precision

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

## Expected Performance

With default configuration on FIVES512:

### UNet (exp001_basic_unet)
| Metric | Value |
|--------|-------|
| Training Time | ~1-2 min/epoch (NVIDIA T4) |
| GPU Memory | ~4-6 GB |
| Model Parameters | ~7.8M |
| Expected Dice | 0.70-0.80+ after 10-20 epochs |
| Expected IoU | 0.60-0.70+ after 10-20 epochs |

### RoiNet (exp002_roinet)
| Metric | Value |
|--------|-------|
| Training Time | ~2-3 min/epoch (NVIDIA T4) |
| GPU Memory | ~6-8 GB |
| Model Parameters | Varies with config |
| Test Dice (epoch 8) | 0.8259 |
| Test IoU (epoch 8) | 0.7169 |
| Validation Dice (epoch 8) | 0.8721 |


---

## Key Design Decisions

1. Configuration-Driven: All parameters in YAML files, no hardcoded values
2. Modular: Each component is independent and replaceable
3. Reproducible: Seed management, config saving, deterministic operations
4. Extensible: Easy to add new models, losses, metrics via registry pattern
5. User-Friendly: Shell scripts for common operations, clear error messages
6. Multiple Architectures: Support for both classic (UNet) and advanced (RoiNet, UTrans, TransRoiNet) models
7. Real-time Visualization: Integrated TensorBoard for training monitoring and experiment comparison

---

## Citation

If you use this pipeline, please cite the FIVES dataset:

```bibtex
@article{fives2022,
  title={FIVES: A Fundus Image Dataset for Vessel Segmentation},
  journal={Scientific Data},
  year={2022}
}
```

---

## Support

For issues, questions, or contributions:
1. Check this README and INSTALL.md first
2. Review your configuration files
3. Check error messages carefully
4. Verify data paths and formats

---