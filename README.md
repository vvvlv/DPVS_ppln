# Vessel Segmentation Pipeline

A clean, modular pipeline for training UNet models on vessel segmentation tasks.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset in this structure:
```
data/FIVES512/
├── train/
│   ├── image/
│   └── label/
├── val/
│   ├── image/
│   └── label/
└── test/
    ├── image/
    └── label/
```

## Training

Using the shell script:
```bash
chmod +x train.sh
./train.sh exp001_basic_unet
```

Or directly with Python:
```bash
python scripts/train.py --config configs/experiments/exp001_basic_unet.yaml
```

## Testing

Test a trained model:
```bash
python scripts/test.py --config outputs/experiments/exp001_basic_unet/config.yaml
```

## Project Structure

```
project/
├── configs/
│   ├── datasets/           # Dataset configurations
│   └── experiments/        # Experiment configurations
├── src/
│   ├── data/              # Dataset and data loading
│   ├── models/            # Model architectures
│   ├── training/          # Training loop and losses
│   └── utils/             # Utilities
├── scripts/               # Training and testing scripts
└── outputs/              # Training outputs (created automatically)
```

## Configuration

All experiment parameters are defined in YAML files under `configs/experiments/`.
See `configs/experiments/exp001_basic_unet.yaml` for an example.

## Output

Training outputs are saved to `outputs/experiments/<experiment_name>/`:
- `config.yaml` - Copy of experiment configuration
- `checkpoints/` - Model checkpoints (best.pth, last.pth)
- `predictions/` - Test set predictions
- `metrics.json` - Final test metrics 