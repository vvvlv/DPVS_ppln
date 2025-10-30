# Advanced TensorBoard Logging Guide

## Overview

This guide explains how to use advanced logging features:
1. **AUC Metric** - Area Under ROC Curve
2. **Activation Logging** - Monitor layer activations with histograms

## 1. AUC Metric Logging

### What is AUC?

AUC (Area Under the ROC Curve) measures the model's ability to distinguish between classes. It's particularly useful for imbalanced datasets and provides a threshold-independent metric.

- **Range**: 0.0 to 1.0
- **Interpretation**: 
  - 0.5 = Random classifier
  - > 0.5 = Better than random
  - 1.0 = Perfect classifier

### Enabling AUC Metric

Simply add `"auc"` to your metrics list:

```yaml
training:
  metrics:
    - "dice"
    - "iou"
    - "auc"      # ⭐ Add this line
```

### Where to Find AUC in TensorBoard

After training starts:
1. Open TensorBoard
2. Go to **Scalars** tab
3. Look for:
   - `train/auc` - Training AUC
   - `val/auc` - Validation AUC
   - `comparison/auc` - Train vs Val comparison

### Example Configuration

```yaml
training:
  epochs: 20
  loss:
    type: "dice"
  metrics:
    - "dice"      # Standard Dice coefficient
    - "iou"       # Intersection over Union
    - "auc"       # Area Under ROC Curve
```

### When to Use AUC

✅ **Use AUC when**:
- Dataset is imbalanced (e.g., small vessels vs background)
- You want threshold-independent evaluation
- Comparing models with different operating points
- Need a single metric for model selection

❌ **Don't use AUC when**:
- Only care about specific threshold (0.5)
- Spatial overlap is more important than pixel-wise classification
- Dice/IoU already provides sufficient information

---

## 2. Activation Logging

### What is Activation Logging?

Activation logging monitors the output of neural network layers during training, helping you:
- **Detect dead neurons** (all zeros)
- **Identify vanishing/exploding gradients** (very small/large values)
- **Monitor layer behavior** over epochs
- **Debug training issues** (layers not learning)

### Enabling Activation Logging

Add to your experiment config:

```yaml
logging:
  tensorboard: true
  log_activations: true              # Enable activation logging
  activation_log_frequency: 5        # Log every 5 epochs (default)
  activation_layers: "auto"          # Use default layers for your model
```

### Configuration Options

#### Option 1: Automatic Layer Selection (Recommended)

```yaml
logging:
  log_activations: true
  activation_layers: "auto"          # Model-specific defaults
```

**Default layers by model**:

**UNet**:
- `encoder1`, `encoder2`, `encoder3`, `encoder4`
- `bottleneck`
- `decoder1`, `decoder2`, `decoder3`, `decoder4`

**RoiNet**:
- `dict_module.conv0` through `dict_module.conv5`
- `dict_module.bottle1`, `dict_module.bottle2`

**UTrans**:
- `encoder1` through `encoder4`
- `bottleneck_conv_in`, `transformer`, `bottleneck_conv_out`
- `decoder1` through `decoder4`

**TransRoiNet**:
- `dict_module.conv0` through `dict_module.conv5`
- `dict_module.bottle1`, `dict_module.transformer`, `dict_module.bottle2`

#### Option 2: Custom Layer Selection

```yaml
logging:
  log_activations: true
  activation_layers:
    - "encoder1"
    - "encoder2"
    - "bottleneck"
    - "decoder1"
```

#### Option 3: Monitor All Layers

```yaml
logging:
  log_activations: true
  activation_layers: null            # Monitor ALL named modules
```

⚠️ **Warning**: Logging all layers can be slow and use significant disk space.

### Logging Frequency

Control how often activations are logged:

```yaml
# Every 5 epochs (default, balanced)
activation_log_frequency: 5

# Every epoch (detailed tracking)
activation_log_frequency: 1

# Every 10 epochs (efficient for long runs)
activation_log_frequency: 10
```

### What Gets Logged

For each monitored layer:

1. **Histogram**: Distribution of activation values
2. **Statistics**:
   - `mean`: Average activation value
   - `std`: Standard deviation
   - `min`: Minimum value
   - `max`: Maximum value

### Viewing in TensorBoard

#### Histograms Tab

1. Open TensorBoard
2. Go to **Histograms** tab
3. See distributions for each layer over time
4. **3D view**: Shows evolution across epochs

Look for:
- ✅ **Healthy**: Bell-shaped curves, evolving over time
- ⚠️ **Dead neurons**: Spike at zero
- ⚠️ **Saturation**: Spikes at extremes (0 or 1 for sigmoid/tanh)
- ⚠️ **Exploding**: Very large values (> 10)
- ⚠️ **Vanishing**: Very small values (< 0.01)

#### Scalars Tab

Navigate to `activations/` section:
- `activations/encoder1/mean`
- `activations/encoder1/std`
- `activations/bottleneck/mean`
- etc.

**Healthy patterns**:
- Mean values relatively stable
- Standard deviation > 0 (layer is learning)
- No sudden jumps or crashes

**Warning signs**:
- Mean approaching 0 → Dead layer
- Std approaching 0 → Not learning
- Mean/Max exploding → Gradient explosion
- All values very small → Vanishing gradients

---

## Complete Example Configuration

```yaml
name: "exp_advanced_logging"
description: "Full logging with AUC and activations"

model:
  type: "UNet"
  # ... model config ...

training:
  epochs: 20
  
  metrics:
    - "dice"
    - "iou"
    - "auc"          # ⭐ AUC metric
  
  # ... other training config ...

logging:
  tensorboard: true
  
  # Image logging
  log_images: true
  image_log_frequency: 5
  
  # ⭐ Activation logging
  log_activations: true
  activation_log_frequency: 5
  activation_layers: "auto"
```

---

## Running an Experiment with Advanced Logging

### Step 1: Use Pre-configured Example

```bash
./train.sh exp_advanced_logging
```

Or create your own:

```bash
# Copy example
cp configs/experiments/exp_advanced_logging.yaml configs/experiments/my_experiment.yaml

# Edit as needed
nano configs/experiments/my_experiment.yaml

# Train
./train.sh my_experiment
```

### Step 2: Launch TensorBoard

```bash
tensorboard --logdir outputs/experiments/my_experiment/tensorboard
```

### Step 3: Monitor in TensorBoard

Open `http://localhost:6006` and check:

1. **Scalars Tab**:
   - Training/validation metrics including AUC
   - Activation statistics per layer

2. **Histograms Tab**:
   - Activation distributions
   - Evolution over epochs (3D view)

3. **Images Tab** (if enabled):
   - Prediction quality visualization

---

## Troubleshooting

### AUC Issues

**"AUC calculation failed"**:
- Usually happens when only one class is present in a batch
- The code handles this gracefully (returns 0.5)
- Increase batch size if this happens frequently

**AUC seems wrong**:
- Ensure predictions are probabilities (0-1), not binary
- Check that `final_activation: "sigmoid"` in model config

### Activation Logging Issues

**"Layer not found in model"**:
- Check layer names with: `print(dict(model.named_modules()).keys())`
- Use correct names (case-sensitive)
- Try `activation_layers: "auto"` first

**TensorBoard histograms not appearing**:
- Wait for first logging epoch
- Check `activation_log_frequency` setting
- Verify `log_activations: true`

**Disk space running out**:
- Reduce `activation_log_frequency` to 10 or higher
- Monitor fewer layers (use custom list)
- Clear old TensorBoard logs

**Training is slow**:
- Activation logging adds ~1-2% overhead per logging epoch
- Increase `activation_log_frequency`
- Reduce number of monitored layers

---

## Performance Impact

### AUC Metric

- **Training Time**: +0.1-0.5% per epoch
- **Memory**: Negligible
- **Disk**: Negligible (~1 KB per epoch)

**Recommendation**: ✅ Always enable if useful for your task

### Activation Logging

- **Training Time**: +1-2% on logging epochs, 0% on other epochs
- **Memory**: +10-50 MB during logging
- **Disk**: ~5-20 MB per logging epoch (depends on layers and frequency)

**Recommendations**:
- ✅ `activation_log_frequency: 5` - Good balance
- ✅ Use `"auto"` for layer selection
- ⚠️ `activation_log_frequency: 1` - Only for debugging (high disk usage)
- ⚠️ `activation_layers: null` - Avoid unless necessary

---

## Best Practices

### For Development/Debugging

```yaml
logging:
  tensorboard: true
  log_images: true
  image_log_frequency: 1           # Every epoch
  log_activations: true
  activation_log_frequency: 1      # Every epoch
  activation_layers: "auto"
```

### For Production Training

```yaml
logging:
  tensorboard: true
  log_images: true
  image_log_frequency: 5           # Every 5 epochs
  log_activations: true
  activation_log_frequency: 10     # Every 10 epochs
  activation_layers: "auto"
```

### For Long Training Runs (50+ epochs)

```yaml
logging:
  tensorboard: true
  log_images: true
  image_log_frequency: 10          # Every 10 epochs
  log_activations: false           # Disable to save space
```

### For Debugging Specific Layers

```yaml
logging:
  tensorboard: true
  log_activations: true
  activation_log_frequency: 1
  activation_layers:                # Only suspicious layers
    - "bottleneck"
    - "encoder4"
```

---

## Example Use Cases

### Use Case 1: Debugging Training Issues

**Problem**: Model not learning, loss stuck

**Solution**:
```yaml
metrics: ["dice", "iou", "auc"]
log_activations: true
activation_log_frequency: 1
```

**What to check**:
1. AUC to verify model is better than random
2. Activation histograms for dead neurons
3. Activation statistics for vanishing gradients

### Use Case 2: Model Comparison

**Problem**: Comparing UNet vs RoiNet performance

**Solution**:
```yaml
# Both experiments with same config
metrics: ["dice", "iou", "auc"]
log_activations: true
activation_layers: "auto"
```

**What to compare**:
- AUC curves in TensorBoard (Scalars tab)
- Activation patterns (which model has healthier activations?)

### Use Case 3: Fine-tuning Learning Rate

**Problem**: Finding optimal learning rate

**Solution**:
```yaml
# Run multiple experiments with different LRs
metrics: ["dice", "iou", "auc"]
log_activations: true
```

**What to check**:
- AUC convergence speed
- Activation statistics stability
- Which LR gives healthiest activation patterns?

---

## Summary

### Quick Reference

| Feature | Config Key | Values | Default |
|---------|-----------|--------|---------|
| **AUC Metric** | `metrics` | Add `"auc"` | Not included |
| **Activation Logging** | `log_activations` | `true`/`false` | `false` |
| **Activation Frequency** | `activation_log_frequency` | Integer > 0 | 5 |
| **Layer Selection** | `activation_layers` | `"auto"`, list, or `null` | Not set |

### TensorBoard Tabs

| Tab | What's There | When to Use |
|-----|--------------|-------------|
| **Scalars** | AUC curves, activation stats | Always |
| **Histograms** | Activation distributions | Debugging, understanding |
| **Images** | Predictions | Quality assessment |
| **Graphs** | Model architecture | Verification |
| **HParams** | Hyperparameters | Experiment comparison |

---

**Ready to leverage advanced logging for better model development!** 🚀

