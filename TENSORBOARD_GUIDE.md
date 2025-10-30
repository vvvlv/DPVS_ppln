# TensorBoard Quick Start Guide

## Overview

TensorBoard is now fully integrated into the training pipeline for real-time visualization and experiment tracking.

## Quick Start

### 1. Start Training (TensorBoard enabled by default)

```bash
./train.sh exp001_basic_unet
```

### 2. Launch TensorBoard (in a separate terminal)

```bash
# For a single experiment
tensorboard --logdir outputs/experiments/exp001_basic_unet/tensorboard

# For all experiments (recommended)
tensorboard --logdir outputs/experiments
```

### 3. Open in Browser

Navigate to: [http://localhost:6006](http://localhost:6006)

## What You'll See

### Scalars Tab

Real-time plots of:
- **Training metrics**: loss, dice, iou
- **Validation metrics**: loss, dice, iou
- **Learning rate**: tracks scheduler changes
- **Comparison plots**: train vs val on same chart

### Images Tab (if `log_images: true`)

Visual comparison of:
- Input images (denormalized)
- Ground truth masks
- Model predictions

Updated every 5 epochs or when best model is saved.

### Graphs Tab

Complete model architecture visualization showing:
- Layer connections
- Tensor flow through the network
- Model structure

### HParams Tab

Compare hyperparameters across experiments:
- Model architecture choices
- Learning rates, batch sizes
- Final and best metrics
- Parallel coordinates visualization

## Configuration

In your experiment config file (e.g., `configs/experiments/exp001_basic_unet.yaml`):

```yaml
logging:
  tensorboard: true           # Enable TensorBoard logging
  log_images: false           # Enable image logging (optional, increases disk usage)
  image_log_frequency: 5      # Log images every N epochs (default: 5, set to 1 for every epoch)
```

## Common Commands

```bash
# Single experiment
tensorboard --logdir outputs/experiments/exp001_basic_unet/tensorboard

# Compare multiple experiments
tensorboard --logdir outputs/experiments

# Use different port
tensorboard --logdir outputs/experiments --port 6007

# On remote server (SSH port forwarding)
ssh -L 6006:localhost:6006 user@server
tensorboard --logdir /path/to/experiments
```

## Image Logging Options

Control how often images are logged to TensorBoard:

```yaml
# Log images every 5 epochs (default - balanced)
logging:
  tensorboard: true
  log_images: true
  image_log_frequency: 5

# Log images every epoch (for detailed evolution tracking)
logging:
  tensorboard: true
  log_images: true
  image_log_frequency: 1

# Log images every 10 epochs (for long training runs)
logging:
  tensorboard: true
  log_images: true
  image_log_frequency: 10
```

**Note**: Images are also logged whenever a new best model is saved, regardless of the frequency setting.

## Tips

1. **Real-time Monitoring**: TensorBoard auto-refreshes every 30 seconds
2. **Smoothing**: Adjust smoothing slider to reduce noise in plots
3. **Toggle Runs**: Click experiment names to show/hide specific runs
4. **Download Data**: Export scalar data as CSV/JSON for further analysis
5. **Pin to Dashboard**: Pin important plots for quick access

## Troubleshooting

### Port Already in Use

```bash
tensorboard --logdir outputs/experiments --port 6007
```

### Remote Server Access

```bash
# On local machine
ssh -L 6006:localhost:6006 user@remote-server

# On remote server
tensorboard --logdir /path/to/experiments
```

Then open `http://localhost:6006` on your local browser.

### Logs Not Appearing

1. Ensure `tensorboard: true` in config
2. Check that training has started
3. Refresh TensorBoard (click refresh button)
4. Verify log directory exists: `outputs/experiments/<exp_name>/tensorboard/`

## File Structure

After training with TensorBoard:

```
outputs/experiments/exp001_basic_unet/
├── checkpoints/
│   ├── best.pth
│   └── last.pth
├── config.yaml
├── metrics_history.yaml
└── tensorboard/              # TensorBoard logs
    ├── events.out.tfevents.* # Event files
    └── test/                 # Test results (if test.py was run)
```

## Advanced Usage

### Compare Specific Experiments

```bash
# Create symbolic links to organize experiments
mkdir -p tensorboard_comparisons/baseline_models
ln -s ../../outputs/experiments/exp001_basic_unet/tensorboard tensorboard_comparisons/baseline_models/unet
ln -s ../../outputs/experiments/exp002_roinet/tensorboard tensorboard_comparisons/baseline_models/roinet

tensorboard --logdir tensorboard_comparisons/baseline_models
```

### Multiple TensorBoard Instances

```bash
# Terminal 1: Baseline models
tensorboard --logdir outputs/experiments/baseline_models --port 6006

# Terminal 2: Transformer models
tensorboard --logdir outputs/experiments/transformer_models --port 6007
```

### Export Data for Papers/Reports

1. View scalar data in TensorBoard
2. Click download button (bottom left of plot)
3. Choose format: CSV or JSON
4. Use in matplotlib/plotly for publication figures

## Viewing Test Results

After running tests:

```bash
./test.sh exp001_basic_unet

# View test logs
tensorboard --logdir outputs/experiments/exp001_basic_unet/tensorboard/test
```

Test metrics are logged to a separate `test/` subdirectory to keep them distinct from training logs.

## Integration Details

The TensorBoard integration logs:

1. **Every Epoch**:
   - Training and validation scalars
   - Learning rate
   - Comparison plots

2. **On Best Model Save** (or every 5 epochs if `log_images: true`):
   - Sample predictions with ground truth

3. **At Training Start**:
   - Model architecture graph

4. **At Training End**:
   - Hyperparameters and final metrics

## Disabling TensorBoard

To disable TensorBoard for an experiment:

```yaml
logging:
  tensorboard: false
  log_images: false
```

The pipeline will work normally with only console output and YAML metrics files.

