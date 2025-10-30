# Quick Reference: Advanced Logging Features

## 🎯 Enable AUC Metric

**In your experiment config** (`configs/experiments/*.yaml`):

```yaml
training:
  metrics:
    - "dice"
    - "iou"
    - "auc"      # ⭐ Add this line
```

**View in TensorBoard**:
- Go to **Scalars** tab
- Look for: `train/auc`, `val/auc`, `comparison/auc`

---

## 🔍 Enable Activation Logging

### Option 1: Auto (Recommended)

```yaml
logging:
  log_activations: true
  activation_log_frequency: 5        # Every 5 epochs
  activation_layers: "auto"          # Model-specific defaults
```

### Option 2: Custom Layers

```yaml
logging:
  log_activations: true
  activation_log_frequency: 5
  activation_layers:
    - "encoder1"
    - "bottleneck"
    - "decoder1"
```

### Option 3: All Layers (Not Recommended)

```yaml
logging:
  log_activations: true
  activation_layers: null            # Monitor everything
```

**View in TensorBoard**:
- Go to **Histograms** tab → See distributions
- Go to **Scalars** tab → `activations/` → See statistics

---

## 📊 Complete Example Config

```yaml
name: "my_experiment"
model:
  type: "UNet"

training:
  epochs: 20
  metrics:
    - "dice"
    - "iou"
    - "auc"                          # ⭐ AUC

logging:
  tensorboard: true
  log_images: true
  image_log_frequency: 5
  log_activations: true              # ⭐ Activations
  activation_log_frequency: 5
  activation_layers: "auto"
```

---

## 🚀 Quick Start

```bash
# Use pre-configured example
./train.sh exp_advanced_logging

# View in TensorBoard
tensorboard --logdir outputs/experiments/exp_advanced_logging/tensorboard
```

---

## 📈 What You'll See

### Scalars Tab
- `train/dice`, `train/iou`, `train/auc`
- `val/dice`, `val/iou`, `val/auc`
- `comparison/dice`, `comparison/iou`, `comparison/auc`
- `activations/encoder1/mean`, `activations/encoder1/std`, etc.

### Histograms Tab
- `activations/encoder1/histogram`
- `activations/encoder2/histogram`
- `activations/bottleneck/histogram`
- etc. (3D view shows evolution)

---

## ⚙️ Frequency Settings

| Use Case | Image Freq | Activation Freq |
|----------|------------|-----------------|
| **Debugging** | 1 | 1 |
| **Development** | 5 | 5 |
| **Production** | 5 | 10 |
| **Long runs** | 10 | 10 or disabled |

---

## 🔧 Troubleshooting

### AUC Not Appearing
✅ Check: `metrics` list includes `"auc"`  
✅ Install: `pip install scikit-learn>=1.3.0`

### Activations Not Appearing
✅ Check: `log_activations: true`  
✅ Check: Waited for first logging epoch  
✅ Verify: Layer names correct (use `"auto"`)

### Disk Space Issues
✅ Increase: `activation_log_frequency` to 10  
✅ Reduce: Number of layers to monitor  
✅ Disable: `log_activations: false`

---

## 📖 Full Documentation

- **Complete Guide**: `ADVANCED_LOGGING_GUIDE.md`
- **TensorBoard Basics**: `TENSORBOARD_GUIDE.md`
- **Implementation**: `TENSORBOARD_IMPLEMENTATION.md`

---

**Level up your model debugging!** 🎉

