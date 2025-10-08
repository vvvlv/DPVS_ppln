# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA 11.8 or higher (for GPU support)
- pip or conda package manager

## Quick Install

### Option 1: Full Installation (Recommended)
Includes optional visualization and analysis tools:

```bash
cd /home/vlv/Documents/master/deepLearning/project/codebase
pip install -r requirements.txt
```


## Verify Installation

Check if PyTorch can access your GPU:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output (with NVIDIA T4):
```
PyTorch: 2.1.0
CUDA available: True
CUDA version: 11.8
GPU: Tesla T4
```

## Package Breakdown

### Core Dependencies (Required)
- **torch** (2.1.0) - Deep learning framework
- **torchvision** (0.16.0) - Vision utilities
- **numpy** (1.24.3) - Numerical computing
- **opencv-python** (4.8.1.78) - Image processing
- **pyyaml** (6.0.1) - Configuration files
- **tqdm** (4.66.1) - Progress bars

### Optional Dependencies
- **matplotlib** (3.8.0) - Plotting and visualization
- **pillow** (10.1.0) - Image handling
- **scikit-image** (0.22.0) - Image processing utilities

### Development Tools (Commented out)
Uncomment in `requirements.txt` if needed:
- **tensorboard** - Training visualization
- **jupyter** - Interactive notebooks
- **ipython** - Enhanced Python shell

## Troubleshooting

### CUDA Version Mismatch

If you have a different CUDA version, install PyTorch accordingly:

**CUDA 11.7:**
```bash
pip install torch==2.1.0+cu117 torchvision==0.16.0+cu117 --index-url https://download.pytorch.org/whl/cu117
```

**CUDA 12.1:**
```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**CPU Only:**
```bash
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

### OpenCV Issues

If opencv-python has conflicts:
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.8.1.78
```

### Memory Issues During Installation

If pip runs out of memory:
```bash
pip install --no-cache-dir -r requirements.txt
```

## Virtual Environment (Recommended)

### Using venv
```bash
# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt

# Deactivate when done
deactivate
```

### Using conda
```bash
# Create conda environment
conda create -n vessel_seg python=3.10

# Activate
conda activate vessel_seg

# Install PyTorch with conda
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other packages
pip install pyyaml==6.0.1 opencv-python==4.8.1.78 tqdm==4.66.1 matplotlib==3.8.0

# Deactivate when done
conda deactivate
```

## Testing Your Installation

Run a quick test:

```bash
python -c "
import torch
import torchvision
import numpy as np
import cv2
import yaml
from tqdm import tqdm

print('✓ All imports successful!')
print(f'  PyTorch: {torch.__version__}')
print(f'  NumPy: {np.__version__}')
print(f'  OpenCV: {cv2.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
"
```

## Next Steps

After successful installation:
1. Read `QUICK_START.md` for usage
2. Prepare your dataset
3. Run your first training: `./train.sh exp001_basic_unet`

## Package Sizes (Approximate)

- torch + torchvision: ~2.5 GB
- numpy: ~15 MB
- opencv-python: ~90 MB
- Other packages: ~50 MB
- **Total**: ~2.7 GB

Ensure you have sufficient disk space! 