# Test Experiment - Full TensorBoard Setup

## Overview

The test experiment has been configured for comprehensive TensorBoard logging with image visualization at every epoch.

## Configuration

**File**: `configs/experiments/test.yaml`

### Logging Settings

```yaml
logging:
  tensorboard: true             # TensorBoard enabled
  log_images: true              # Image logging enabled
  image_log_frequency: 1        # Log images EVERY epoch (not every 5)
```

### Experiment Details

- **Model**: UNet (baseline)
- **Epochs**: 5 (quick test)
- **Batch Size**: 6
- **Learning Rate**: 0.0001
- **Metrics**: Dice coefficient, IoU

## What Will Be Logged

### Every Epoch (Epochs 1, 2, 3, 4, 5):

1. **Training Metrics**:
   - `train/loss`
   - `train/dice`
   - `train/iou`

2. **Validation Metrics**:
   - `val/loss`
   - `val/dice`
   - `val/iou`

3. **Learning Rate**:
   - `learning_rate` (tracks cosine annealing scheduler)

4. **Comparison Plots**:
   - `comparison/dice` (train vs val)
   - `comparison/iou` (train vs val)

5. **Images** ⭐ (NEW - Every Epoch!):
   - `val/predictions`
   - Shows 4 validation samples:
     - Column 1: Input image (denormalized)
     - Column 2: Ground truth mask
     - Column 3: Model prediction
   - **Total**: 4 rows × 3 columns = 12 images per epoch

### At Training Start:

- **Model Architecture Graph**: Full UNet structure visualization

### At Training End:

- **Hyperparameters**: Complete experiment configuration
- **Final Metrics**: Best validation Dice and IoU scores

## Running the Test Experiment

### 1. Start Training

```bash
cd /home/vlv/Documents/master/deepLearning/project/codebase
./train.sh test
```

### 2. Launch TensorBoard (separate terminal)

```bash
tensorboard --logdir outputs/experiments/test/tensorboard
```

### 3. Open Browser

Navigate to: [http://localhost:6006](http://localhost:6006)

### 4. What to Expect

**Scalars Tab**:
- 6 main plots (train/val for loss, dice, iou)
- 2 comparison plots (dice, iou showing train vs val)
- 1 learning rate plot
- All updating after each epoch

**Images Tab**:
- **5 entries** (one per epoch: 1, 2, 3, 4, 5)
- Each entry shows a 4×3 grid:
  ```
  [Image1] [GT1] [Pred1]
  [Image2] [GT2] [Pred2]
  [Image3] [GT3] [Pred3]
  [Image4] [GT4] [Pred4]
  ```
- Click through epochs to see prediction evolution
- Slider allows you to step through epochs

**Graphs Tab**:
- Complete UNet architecture
- Shows encoder, bottleneck, decoder, skip connections

**HParams Tab** (after training completes):
- All hyperparameters
- Final and best metrics

## Image Evolution Tracking

The key feature for this experiment is **per-epoch image logging**, which allows you to:

1. **Track Learning Progress**: See how predictions improve from epoch 1 to 5
2. **Identify Issues Early**: Spot problems like:
   - Not learning (predictions stay blank)
   - Overfitting (validation predictions get worse)
   - Mode collapse (predictions look identical)
3. **Visual Convergence**: See when the model has "learned" the task
4. **Compare Samples**: See which validation samples are easy/hard

## Expected Behavior

### Epoch 1:
- Predictions will be noisy/poor
- High loss values
- Low Dice/IoU scores

### Epoch 2-3:
- Predictions start to resemble vessels
- Loss decreasing
- Metrics improving

### Epoch 4-5:
- Predictions should be quite good
- Metrics stabilizing
- Clear vessel structures

## File Structure After Training

```
outputs/experiments/test/
├── checkpoints/
│   ├── best.pth              # Best model checkpoint
│   └── last.pth              # Last epoch checkpoint
├── config.yaml               # Experiment configuration
├── metrics_history.yaml      # All epoch metrics in YAML
└── tensorboard/              # TensorBoard logs
    └── events.out.tfevents.* # Event file with all logged data
```

## Disk Usage Estimate

With `image_log_frequency: 1` for 5 epochs:
- **Scalars**: ~100 KB
- **Images**: ~15-25 MB (5 epochs × 4 samples × 3 views)
- **Graph**: ~500 KB
- **Total**: ~20-30 MB

This is acceptable for analysis. For longer runs (20+ epochs), consider `image_log_frequency: 5`.

## Comparing with Other Experiments

After running the test experiment, you can compare it with others:

```bash
# Run a comparison experiment without per-epoch images
./train.sh exp001_basic_unet  # Has image_log_frequency: 5 (default)

# Compare both in TensorBoard
tensorboard --logdir outputs/experiments
```

You'll see:
- **test**: 5 image entries (epochs 1, 2, 3, 4, 5)
- **exp001_basic_unet**: Images only at epochs 5, 10, 15, 20 (and when best model saved)

## New Feature: Configurable Image Logging

This setup introduces a new configuration parameter:

### `image_log_frequency`

**Purpose**: Control how often images are logged to TensorBoard

**Values**:
- `1`: Log every epoch (most detailed, higher disk usage)
- `5`: Log every 5 epochs (default, balanced)
- `10`: Log every 10 epochs (for long training runs)
- Any positive integer

**Default**: `5` (if not specified)

**Example Usage**:

```yaml
# Detailed tracking (short experiments)
logging:
  tensorboard: true
  log_images: true
  image_log_frequency: 1

# Balanced (medium experiments)
logging:
  tensorboard: true
  log_images: true
  image_log_frequency: 5

# Efficient (long experiments)
logging:
  tensorboard: true
  log_images: true
  image_log_frequency: 10
```

**Note**: Images are ALWAYS logged when a new best model is saved, regardless of the frequency setting.

## Tips for Analysis

1. **Use the Slider**: In the Images tab, use the step slider to animate through epochs
2. **Focus on Specific Samples**: Identify which sample (1-4) is hardest to segment
3. **Check for Overfitting**: Compare image quality with validation metrics
4. **Look for Patterns**: See if certain vessel types appear first
5. **Download**: Use TensorBoard's download feature to export metrics for papers

## Troubleshooting

### Images Not Appearing

1. Check config: `log_images: true` and `image_log_frequency: 1`
2. Verify training started (wait for at least 1 full epoch)
3. Refresh TensorBoard (click refresh icon)
4. Check Images tab (not Scalars tab)

### TensorBoard Not Starting

```bash
# Check if port is in use
lsof -i :6006

# Use different port
tensorboard --logdir outputs/experiments/test/tensorboard --port 6007
```

### Disk Space

If disk space is a concern:
```yaml
# Reduce to every 2-3 epochs
image_log_frequency: 2  # or 3
```

Or reduce number of samples (requires code change in trainer.py):
```python
# In _log_sample_images method, change:
max_images=2  # instead of 4
```

## Summary

✅ **Test experiment configured for maximum TensorBoard visibility**  
✅ **Images logged every epoch (5 total)**  
✅ **All metrics tracked**  
✅ **Model graph visualized**  
✅ **Ready to track prediction evolution**  

**Next**: Run `./train.sh test` and watch your model learn in real-time! 🚀

