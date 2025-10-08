"""Testing/Inference script for evaluating trained models."""

import sys
import argparse
from pathlib import Path
import json
import yaml
import torch
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import load_config
from data import create_dataloaders
from models import create_model
from training.metrics import create_metrics


def test(config_path: str, checkpoint_path: str = None, output_dir: str = None):
    """Run inference and testing on test set.
    
    Args:
        config_path: Path to experiment config file
        checkpoint_path: Path to checkpoint file (default: best.pth from experiment)
        output_dir: Output directory for predictions (default: outputs/tests/<experiment_name>)
    """
    
    # Load config
    config = load_config(config_path)
    experiment_name = config['name']
    
    print(f"\n{'='*60}")
    print(f"Testing: {experiment_name}")
    print(f"{'='*60}")
    
    # Create model
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = create_model(config)
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(config['output']['dir']) / 'checkpoints' / 'best.pth'
    
    checkpoint_path = Path(checkpoint_path)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create test loader
    _, _, test_loader = create_dataloaders(config)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create metrics
    metrics = create_metrics(config['training']['metrics'])
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path('outputs/tests') / experiment_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(exist_ok=True)
    
    print(f"Saving predictions to: {pred_dir}")
    
    # Test
    metric_sums = {name: 0 for name in config['training']['metrics']}
    per_image_metrics = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device) if isinstance(batch['image'], torch.Tensor) else torch.from_numpy(batch['image']).to(device)
            masks = batch['mask'].to(device) if isinstance(batch['mask'], torch.Tensor) else torch.from_numpy(batch['mask']).to(device)
            names = batch['name']
            
            # Run inference
            outputs = model(images)
            
            # Calculate metrics for each image
            for i, name in enumerate(names):
                image_metrics = {'image': name}
                
                for metric_name, metric_fn in metrics.items():
                    # Calculate metric for single image (already returns float)
                    metric_value = metric_fn(outputs[i:i+1], masks[i:i+1])
                    image_metrics[metric_name] = float(metric_value)
                    metric_sums[metric_name] += metric_value
                
                per_image_metrics.append(image_metrics)
                
                # Save prediction
                pred = outputs[i, 0].cpu().numpy()
                pred_binary = (pred > 0.5).astype(np.uint8) * 255
                cv2.imwrite(str(pred_dir / name), pred_binary)
    
    # Calculate average metrics
    n = len(test_loader.dataset)
    avg_metrics = {k: float(v / len(test_loader)) for k, v in metric_sums.items()}
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for name, value in avg_metrics.items():
        print(f"  {name.upper()}: {value:.4f}")
    print("=" * 60)
    
    # Save average metrics
    metrics_summary = {
        'experiment': experiment_name,
        'checkpoint': str(checkpoint_path),
        'num_test_images': n,
        'average_metrics': avg_metrics
    }
    
    summary_path = output_dir / 'test_metrics.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(metrics_summary, f, default_flow_style=False, sort_keys=False)
    print(f"\nMetrics summary saved to: {summary_path}")
    
    # Save per-image metrics
    per_image_path = output_dir / 'per_image_metrics.yaml'
    with open(per_image_path, 'w') as f:
        yaml.dump(per_image_metrics, f, default_flow_style=False, sort_keys=False)
    print(f"Per-image metrics saved to: {per_image_path}")
    
    print(f"\nPredicted masks saved to: {pred_dir}")
    print(f"\n✓ Testing complete!")
    
    return avg_metrics, per_image_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference and evaluate model on test set')
    parser.add_argument('--config', required=True, help='Path to experiment config file')
    parser.add_argument('--checkpoint', default=None, help='Path to checkpoint (default: best.pth from experiment)')
    parser.add_argument('--output-dir', default=None, help='Output directory for predictions (default: outputs/tests/<exp_name>)')
    args = parser.parse_args()
    
    test(args.config, args.checkpoint, args.output_dir) 