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
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    model = create_model(config)
    print(f"  Model: {config['model']['type']}")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Device: {device}")
    
    # Create trainer and train
    trainer = Trainer(model, train_loader, val_loader, config, device)
    trainer.train()
    
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