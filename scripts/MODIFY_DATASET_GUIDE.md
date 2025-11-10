# Script to modify FIVES Dataset

- to resize the FIVES dataset into multiple sizes
- to convert samples into multiple channels 

### You can rezise samples to multiple target sizes 
- Maximum supported image size is 1024x1024
- For example:
    - 256x256
    - 512x512
    - 512x256

### You can convert samples to multiple channel configurations
- RGB (3 channels)
- Grayscale (1 channel)
- Single color channels
    - red
    - green
    - blue

## Usage

```
python modify_dataset.py \
  --input <input_dataset_path> \
  --output <output_path> \
  --sizes <size1> <size2> ... \
  --channels <channel1> <channel2> ...
```

Example:
```
python modify_dataset.py \
  --input path/to/FIVES \
  --output path/to/output \
  --sizes 256 512 \
  --channels rgb green
```

## Output Structure
```
output_dir/
├── FIVES_256x256_rgb/
│   ├── train/
│   │   ├── image/
│   │   └── label/
│   ├── val/
│   │   ├── image/
│   │   └── label/
│   └── test/
│       ├── image/
│       └── label/
└── FIVES_512x512_green/
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