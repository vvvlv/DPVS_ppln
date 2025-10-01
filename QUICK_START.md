# Quick Start Guide

## ✅ What's Been Implemented

A **minimal, working UNet pipeline** for vessel segmentation with:

- ✓ Clean configuration system (Dataset + Experiment YAMLs)
- ✓ Dataset loader with automatic normalization
- ✓ Basic UNet architecture (5 levels: 32→64→128→256→512)
- ✓ Dice loss function
- ✓ Dice coefficient & IoU metrics
- ✓ Training loop with validation
- ✓ Early stopping & checkpointing
- ✓ Testing script with predictions
- ✓ Shell launcher for easy execution

## 📁 Setup Your Data

Place your FIVES512 dataset here:
```
/home/vlv/Documents/master/deepLearning/project/codebase/data/FIVES512/
├── train/
│   ├── image/ (*.png)
│   └── label/ (*.png)
├── val/
│   ├── image/
│   └── label/
└── test/
    ├── image/
    └── label/
```

Or update the path in `configs/datasets/fives_512.yaml`

## 🚀 Train Your First Model

```bash
# Option 1: Using the shell script
./train.sh exp001_basic_unet

# Option 2: Direct Python
python scripts/train.py --config configs/experiments/exp001_basic_unet.yaml
```

## 🧪 Test the Model

```bash
python scripts/test.py --config outputs/experiments/exp001_basic_unet/config.yaml
```

This will:
- Load the best checkpoint
- Run inference on test set
- Calculate metrics (Dice, IoU)
- Save predictions to `outputs/experiments/exp001_basic_unet/predictions/`
- Save metrics to `outputs/experiments/exp001_basic_unet/metrics.json`

## 📊 Output Structure

After training, you'll find:
```
outputs/experiments/exp001_basic_unet/
├── config.yaml              # Copy of experiment config
├── checkpoints/
│   ├── best.pth            # Best validation model
│   └── last.pth            # Latest checkpoint
├── predictions/            # Test predictions (after running test.py)
└── metrics.json           # Final test metrics
```

## ⚙️ Modify the Configuration

Edit `configs/experiments/exp001_basic_unet.yaml`:

**Quick tweaks:**
- `training.epochs: 10` → Change number of epochs
- `data.batch_size: 4` → Adjust batch size for your GPU
- `model.depths: [32, 64, 128, 256, 512]` → Make model smaller/larger
- `training.learning_rate: 0.0001` → Tune learning rate

**Create a new experiment:**
```bash
cp configs/experiments/exp001_basic_unet.yaml configs/experiments/exp002_my_test.yaml
# Edit exp002_my_test.yaml
./train.sh exp002_my_test
```

## 🔧 Current Limitations (By Design)

This is the **minimal working version**. Not yet implemented:
- ❌ Data augmentation (structure ready, not active)
- ❌ Attention mechanisms (skeleton in blocks/)
- ❌ Transformer blocks (skeleton in blocks/)
- ❌ Mixed precision training (flag exists but not active)
- ❌ TensorBoard logging (flag exists but not implemented)
- ❌ Additional loss functions (only Dice)

These will be added later following the pipeline structure!

## 🐛 Troubleshooting

**"No images found in..."**
- Check your data path in `configs/datasets/fives_512.yaml`
- Ensure images are `.png` format

**CUDA out of memory**
- Reduce `batch_size` in experiment config
- Reduce model size: `depths: [16, 32, 64, 128, 256]`

**Import errors**
- Run from the codebase root: `cd /home/vlv/Documents/master/deepLearning/project/codebase`
- Check all `__init__.py` files exist

## 📝 Next Steps

To extend the pipeline:

1. **Add data augmentation** → Edit `src/data/transforms.py`
2. **Add new loss functions** → Add to `src/training/losses.py`
3. **Add attention/transformers** → Use skeletons in `src/models/blocks/`
4. **Add TensorBoard** → Create logger in `src/utils/logger.py`

All structures are already in place per the specification! 