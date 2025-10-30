"""Main training script."""

import sys
import argparse
from pathlib import Path
import yaml
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import load_config
from utils.helpers import set_seed, count_parameters
from utils.tensorboard_logger import TensorBoardLogger
from data import create_dataloaders
from models import create_model
from training.trainer import Trainer


def main(config_path: str):
    """Main training function."""
    
    # Load configuration
    config = load_config(config_path)
    
    print("=" * 70)
    print(f"Experiment: {config['name']}")
    print(f"Description: {config['description']}")
    print("=" * 70)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Create output directory and save config
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Create data loaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    if config['device'].startswith('cuda') and torch.cuda.is_available():
        device = torch.device(config['device'])
    elif config['device'].startswith('mps') and getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = create_model(config)
    print(f"  Model: {config['model']['type']}")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Device: {device}")
    
    # Initialize TensorBoard logger if enabled
    tb_logger = None
    if config.get('logging', {}).get('tensorboard', False):
        tb_log_dir = output_dir / 'tensorboard'
        tb_logger = TensorBoardLogger(str(tb_log_dir), enabled=True)
    
    # Create trainer and train
    try:
        trainer = Trainer(model, train_loader, val_loader, config, device, tb_logger)
        trainer.train()
    finally:
        # Ensure TensorBoard logger is properly closed
        if tb_logger is not None:
            tb_logger.close()
    
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train vessel segmentation model')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to experiment config file'
    )
    args = parser.parse_args()
    
    main(args.config)