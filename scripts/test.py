"""Testing script for evaluating trained models."""

import sys
import argparse
from pathlib import Path
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import load_config
from data import create_dataloaders
from models import create_model
from training.metrics import create_metrics


def test(config_path: str, checkpoint_path: str = None):
    """Run testing on test set."""
    
    # Load config
    config = load_config(config_path)
    
    print(f"Testing: {config['name']}")
    
    # Create model
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    model = create_model(config)
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(config['output']['dir']) / 'checkpoints' / 'best.pth'
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create test loader
    _, _, test_loader = create_dataloaders(config)
    
    # Create metrics
    metrics = create_metrics(config['training']['metrics'])
    
    # Test
    metric_sums = {name: 0 for name in config['training']['metrics']}
    
    # Output directory for predictions
    pred_dir = Path(config['output']['dir']) / 'predictions'
    if config['output']['save_predictions']:
        pred_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = torch.from_numpy(batch['image']).to(device)
            masks = torch.from_numpy(batch['mask']).to(device)
            names = batch['name']
            
            outputs = model(images)
            
            for metric_name, metric_fn in metrics.items():
                metric_sums[metric_name] += metric_fn(outputs, masks)
            
            # Save predictions
            if config['output']['save_predictions']:
                for i, name in enumerate(names):
                    pred = outputs[i, 0].cpu().numpy()
                    pred = (pred > 0.5).astype(np.uint8) * 255
                    cv2.imwrite(str(pred_dir / name), pred)
    
    # Average
    n = len(test_loader)
    results = {k: v / n for k, v in metric_sums.items()}
    
    # Print
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    for name, value in results.items():
        print(f"  {name}: {value:.4f}")
    print("=" * 50)
    
    # Save
    results_path = Path(config['output']['dir']) / 'metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config')
    parser.add_argument('--checkpoint', default=None, help='Path to checkpoint')
    args = parser.parse_args()
    
    test(args.config, args.checkpoint) 