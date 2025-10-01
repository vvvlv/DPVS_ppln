"""Main training loop handler."""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import Dict


class Trainer:
    """Training loop manager."""
    
    def __init__(self, model: nn.Module, train_loader, val_loader, 
                 config: dict, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_loss()
        self.metrics = self._create_metrics()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.early_stop_counter = 0
        
        # Create output directory
        self.output_dir = Path(config['output']['dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['epochs']
        
        print(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Print metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, " +
                  ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items() if k != 'loss']))
            print(f"  Val   - Loss: {val_metrics['val_loss']:.4f}, " +
                  ", ".join([f"{k.replace('val_', '')}: {v:.4f}" for k, v in val_metrics.items() if k != 'val_loss']))
            
            # Checkpointing
            self._handle_checkpointing(val_metrics)
            
            # Early stopping
            if self._check_early_stopping(val_metrics):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Step scheduler
            self.scheduler.step()
        
        print("\n✓ Training complete!")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        metric_sums = {name: 0 for name in self.config['training']['metrics']}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch in pbar:
            images = batch['image'].to(self.device) if isinstance(batch['image'], torch.Tensor) else torch.from_numpy(batch['image']).to(self.device)
            masks = batch['mask'].to(self.device) if isinstance(batch['mask'], torch.Tensor) else torch.from_numpy(batch['mask']).to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward
            loss.backward()
            
            if self.config['training'].get('gradient_clip'):
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            
            with torch.no_grad():
                for metric_name, metric_fn in self.metrics.items():
                    metric_sums[metric_name] += metric_fn(outputs, masks)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average
        n = len(self.train_loader)
        return {
            'loss': total_loss / n,
            **{k: v / n for k, v in metric_sums.items()}
        }
    
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0
        metric_sums = {name: 0 for name in self.config['training']['metrics']}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]"):
                images = batch['image'].to(self.device) if isinstance(batch['image'], torch.Tensor) else torch.from_numpy(batch['image']).to(self.device)
                masks = batch['mask'].to(self.device) if isinstance(batch['mask'], torch.Tensor) else torch.from_numpy(batch['mask']).to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                
                for metric_name, metric_fn in self.metrics.items():
                    metric_sums[metric_name] += metric_fn(outputs, masks)
        
        n = len(self.val_loader)
        return {
            'val_loss': total_loss / n,
            **{f'val_{k}': v / n for k, v in metric_sums.items()}
        }
    
    def _create_optimizer(self):
        """Create optimizer from config."""
        opt_cfg = self.config['training']['optimizer']
        
        if opt_cfg['type'] == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=opt_cfg['learning_rate'],
                weight_decay=opt_cfg['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg['type']}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        sched_cfg = self.config['training']['scheduler']
        
        if sched_cfg['type'] == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_cfg['min_lr']
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_cfg['type']}")
    
    def _create_loss(self):
        """Create loss function."""
        from .losses import create_loss
        return create_loss(self.config['training']['loss'])
    
    def _create_metrics(self):
        """Create metric functions."""
        from .metrics import create_metrics
        return create_metrics(self.config['training']['metrics'])
    
    def _handle_checkpointing(self, metrics: Dict[str, float]):
        """Save checkpoints."""
        ckpt_cfg = self.config['training']['checkpoint']
        metric_name = ckpt_cfg['monitor']
        current_metric = metrics[metric_name]
        
        # Save best
        if ckpt_cfg['save_best'] and current_metric > self.best_metric:
            self.best_metric = current_metric
            self._save_checkpoint('best.pth')
            print(f"  → New best {metric_name}: {current_metric:.4f}")
        
        # Save last
        if ckpt_cfg['save_last']:
            self._save_checkpoint('last.pth')
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        ckpt_dir = self.output_dir / 'checkpoints'
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric
        }, ckpt_dir / filename)
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria met."""
        es_cfg = self.config['training']['early_stopping']
        if not es_cfg['enabled']:
            return False
        
        metric_name = es_cfg['monitor']
        current_metric = metrics[metric_name]
        
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.early_stop_counter = 0
            return False
        else:
            self.early_stop_counter += 1
            return self.early_stop_counter >= es_cfg['patience'] 