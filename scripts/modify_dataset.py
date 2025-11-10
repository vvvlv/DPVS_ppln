"""
Resize FIVES dataset to multiple sizes and channels.
"""

from typing import List, Tuple
from pathlib import Path
from PIL import Image
import argparse
import sys

def parse_size(size: str, max_size: int = 1024) -> Tuple[int, int]:
    """ Parse unit H and W (NxN) with '256' -> (256,256) OR diverse values HxW like '320x256' -> (320,256). """
    size = size.lower().strip()
    
    try:
        if 'x' in size:
            height, width = size.split('x')
            height, width = int(height), int(width)
        height_x_width = int(size)
        height, width = height_x_width, height_x_width
    except ValueError:
        raise ValueError(f"🙉🙉🙉 [ERROR] Invalid size format '{size}'. Expected either 'N' OR 'HxW', e.g., '256' or '320x256'.")
    
    # --- sanity checks ---
    if height <= 0 or width <= 0:
        raise ValueError(f"🙉🙉🙉 [ERROR] Invalid size '{height}x{width}': dimensions must be positive integers, not 0.")
    if height > max_size or width > max_size:
        raise ValueError(f"🙉🙉🙉 [ERROR] Size {height}x{width} too large; maximum side length is {max_size}.")
    
    return height, width

def find_data(dir: Path) -> List[Path]:
    """ Find all PNG files in directory """
    files = []
    try:
        if not dir.exists():
            return files
        for path in dir.rglob('*.png'):
            if path.is_file():
                files.append(path)
        return files
    except Exception as e:
        print(f"🙉🙉🙉 [ERROR] Cannot find directory {dir}: {e}")

def convert_image_channel(image: Image.Image, channel: str) -> Image.Image:
    """ Convert image to desired channel mode. for example 'rgb' or 'gray'. """
    if channel == 'rgb':
        return image.convert('RGB')
    elif channel == 'gray':
        return image.convert('L')
    elif channel == 'red':
        r, g, b = image.split()
        return r
    elif channel == 'green':
        r, g, b = image.split()
        return g
    elif channel == 'blue':
        r, g, b = image.split()
        return b
    else:
        raise ValueError(f"🙉🙉🙉 [ERROR] Unknown image channel: {channel}")
    
def split(
    input_dir: Path,
    output_dir: Path,
    size: Tuple[int, int],
    channel: str,
    folder: str,
):
    """ Process images and labels for a specific split/folder (train, val, test). """
    
    # input paths
    input_image_dir = input_dir / folder / 'image'
    input_label_dir = input_dir / folder / 'label'
    
    # output paths
    output_image_dir = output_dir / folder / 'image'
    output_label_dir = output_dir / folder / 'label'
    
    # create output directories
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = find_data(input_image_dir)
    print(f"Processing {len(image_paths)} images in '{folder}' for size {size} and channel '{channel}'.")
    
    for img_path in image_paths: 
        relative_path = img_path.relative_to(input_image_dir)
        label_path = input_label_dir / relative_path
        
        image = Image.open(img_path)
        label = Image.open(label_path)
        
        # BILINEAR for image, to keep the quality
        img_resized = image.resize(size, Image.BILINEAR)
        # NEAREST for binary
        label_resized = label.resize(size, Image.NEAREST)
        
        # convert with specified channel
        img_converted = convert_image_channel(img_resized, channel)
        
        # Save outputs
        output_img_path = output_image_dir / relative_path
        output_label_path = output_label_dir / relative_path
        
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        output_label_path.parent.mkdir(parents=True, exist_ok=True)
        
        img_converted.save(output_img_path)
        label_resized.save(output_label_path)
    
def process_dataset(
    input_dir: Path,
    output_dir: Path,
    size: Tuple[int, int],
    channel: str,
):
    """ Create images and labels for specifiesed size and channel. """

    height, width = size
    sample_name = f"FIVES_{height}x{width}_{channel}"
    output_sample_dir = output_dir / sample_name

    for folder in ['train', 'val', 'test']:
        split(input_dir, output_sample_dir, size, channel, folder)


def main():
    args_parse = argparse.ArgumentParser(
                    description="Resize FIVES (images + labels) to multiple sizes and channels.")
    args_parse.add_argument("--input", required=True, type=Path, 
                    help="Input dataset directory.")
    args_parse.add_argument("--output", required=True, type=Path, 
                    help="Output directory for modified images and labels.")
    args_parse.add_argument("--sizes", nargs="+", required=True,
                    help="Target sizes: e.g. 256 512x512 320x256 (N or HxW). Multiple allowed.")
    args_parse.add_argument("--channels", nargs="+", default=["rgb"], 
                    choices=["rgb","gray","red","green","blue"],
                    help="Channel modes to generate.")
    
    args = args_parse.parse_args()

    input_dir  = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory  : {input_dir}")
    print(f"Output directory : {output_dir}")
    print(f"Sizes            : {args.sizes}")
    print(f"Channels         : {args.channels}")

    # Parse sizes once
    sizes = [parse_size(size) for size in args.sizes]

    for H_x_W in sizes:
        for channel in args.channels:
            process_dataset(
                input_dir=input_dir,
                output_dir=output_dir,
                size=H_x_W,
                channel=channel,
            )
    
    print("\nDone yaay! ✅")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted run.")
        sys.exit(1)
