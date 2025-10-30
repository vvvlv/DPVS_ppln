# Quick Start: Test Experiment with Full TensorBoard

## ⚡ Run in 3 Steps

### 1️⃣ Start Training
```bash
cd /home/vlv/Documents/master/deepLearning/project/codebase
./train.sh test
```

### 2️⃣ Launch TensorBoard (new terminal)
```bash
tensorboard --logdir outputs/experiments/test/tensorboard
```

### 3️⃣ Open Browser
Navigate to: **http://localhost:6006**

---

## 📊 What You'll See

### Scalars Tab
- **9 plots** updating after each epoch:
  - Training: loss, dice, iou
  - Validation: loss, dice, iou
  - Learning rate
  - Comparison plots: dice, iou

### Images Tab ⭐ NEW!
- **5 image grids** (one per epoch)
- Each grid shows **4 samples**:
  ```
  [Input Image] [Ground Truth] [Prediction]
  [Input Image] [Ground Truth] [Prediction]
  [Input Image] [Ground Truth] [Prediction]
  [Input Image] [Ground Truth] [Prediction]
  ```
- **Use slider** to step through epochs and watch model learn!

### Graphs Tab
- Full UNet architecture visualization

### HParams Tab
- Hyperparameters and final metrics (after training completes)

---

## 🎯 Key Feature

**Image logging every epoch** (`image_log_frequency: 1`)

This allows you to:
- ✅ See prediction evolution from epoch 1 → 5
- ✅ Identify learning issues early
- ✅ Visually track convergence
- ✅ Compare which samples are hard/easy

---

## ⚙️ Configuration Used

**File**: `configs/experiments/test.yaml`

```yaml
logging:
  tensorboard: true           # ✅ TensorBoard enabled
  log_images: true            # ✅ Image logging enabled  
  image_log_frequency: 1      # ⭐ Every epoch (not every 5)
```

---

## 📁 Output

After training, find logs in:
```
outputs/experiments/test/tensorboard/
```

---

## 🔧 Customization

Want different image frequency?

```yaml
# Every 2 epochs
image_log_frequency: 2

# Every 5 epochs (default for other experiments)
image_log_frequency: 5

# Every 10 epochs (for long runs)
image_log_frequency: 10
```

---

## 💡 Pro Tips

1. **Animate Evolution**: Use TensorBoard's step slider in Images tab
2. **Compare Experiments**: Point tensorboard to `outputs/experiments` to see all runs
3. **Remote Training**: Use SSH port forwarding: `ssh -L 6006:localhost:6006 user@server`
4. **Export Plots**: Click download button in TensorBoard to save as CSV/PNG

---

## 📖 Full Documentation

- **Complete Guide**: See `TENSORBOARD_GUIDE.md`
- **Test Details**: See `TEST_EXPERIMENT_SETUP.md`
- **Implementation**: See `TENSORBOARD_IMPLEMENTATION.md`

---

**Ready to track your model's learning visually!** 🚀

