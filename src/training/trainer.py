"""Main training loop handler."""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional

from utils.tensorboard_logger import (
    ActivationLogger, 
    get_default_layer_names,
    log_activations_to_tensorboard
)


class Trainer:
    """Training loop manager."""
    
    def __init__(self, model: nn.Module, train_loader, val_loader, 
                 config: dict, device, tensorboard_logger=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.tb_logger = tensorboard_logger
        
        # Setup components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_loss()
        self.metrics = self._create_metrics()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.early_stop_counter = 0
        self.metrics_history: List[Dict] = []
        
        # Create output directory
        self.output_dir = Path(config['output']['dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        
        # Setup activation logging (if enabled)
        self.activation_logger = None
        if self.tb_logger is not None and config['logging'].get('log_activations', False):
            # Get layer names to monitor
            layer_names = config['logging'].get('activation_layers', None)
            if layer_names is None or layer_names == 'auto':
                # Use default layers based on model type
                model_type = config['model']['type']
                layer_names = get_default_layer_names(model_type)
                if not layer_names:
                    print(f"Warning: No default activation layers for model '{model_type}'. Monitoring all layers.")
                    layer_names = None
            
            self.activation_logger = ActivationLogger(model, layer_names)
            print(f"Activation logging enabled for {len(layer_names) if layer_names else 'all'} layers")
        
        # Log model graph to TensorBoard (if enabled)
        if self.tb_logger is not None:
            try:
                # Get input size from dataset config
                dataset_config = config.get('dataset', {})
                image_size = dataset_config.get('image_size', [512, 512])
                in_channels = dataset_config.get('num_channels', 3)
                batch_size = config['data'].get('batch_size', 1)
                
                input_size = (batch_size, in_channels, image_size[0], image_size[1])
                self.tb_logger.log_model_graph(model, input_size, str(device))
            except Exception as e:
                print(f"Warning: Could not log model graph: {e}")
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['epochs']
        
        print(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print("\n")
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, " +
                  ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items() if k != 'loss']))
            print(f"  Val   - Loss: {val_metrics['val_loss']:.4f}, " +
                  ", ".join([f"{k.replace('val_', '')}: {v:.4f}" for k, v in val_metrics.items() if k != 'val_loss']))
            
            # Log to TensorBoard
            if self.tb_logger is not None:
                # Log training metrics
                self.tb_logger.log_metrics(train_metrics, 'train', epoch + 1)
                
                # Log validation metrics
                self.tb_logger.log_metrics(val_metrics, 'val', epoch + 1)
                
                # Log learning rate
                self.tb_logger.log_learning_rate(current_lr, epoch + 1)
                
                # Log comparison plots (train vs val for same metrics)
                for metric_name in self.config['training']['metrics']:
                    if metric_name in train_metrics and f'val_{metric_name}' in val_metrics:
                        self.tb_logger.log_scalars(
                            f'comparison/{metric_name}',
                            {
                                'train': train_metrics[metric_name],
                                'val': val_metrics[f'val_{metric_name}']
                            },
                            epoch + 1
                        )
                
                # Flush to ensure data is written
                self.tb_logger.flush()
            
            # Save metrics history
            epoch_metrics = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics
            }
            self.metrics_history.append(epoch_metrics)
            self._save_metrics_history()
            
            # Checkpointing
            self._handle_checkpointing(val_metrics)
            
            # Log images to TensorBoard (if enabled and on specific epochs)
            if self.tb_logger is not None and self.config['logging'].get('log_images', False):
                # Get image logging frequency from config (default: every 5 epochs)
                image_log_freq = self.config['logging'].get('image_log_frequency', 5)
                # Log images based on frequency or when best model is saved
                if (epoch + 1) % image_log_freq == 0 or val_metrics[self.config['training']['checkpoint']['monitor']] >= self.best_metric:
                    self._log_sample_images(epoch + 1)
            
            # Log activations to TensorBoard (if enabled)
            if self.tb_logger is not None and self.activation_logger is not None:
                activation_log_freq = self.config['logging'].get('activation_log_frequency', 5)
                if (epoch + 1) % activation_log_freq == 0:
                    self._log_activations(epoch + 1)
            
            # Early stopping
            if self._check_early_stopping(val_metrics):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Step scheduler
            self.scheduler.step()
        
        # Log final hyperparameters and metrics to TensorBoard
        if self.tb_logger is not None:
            self._log_hyperparameters()
        
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
        
        # Check if metric improved (best_metric is updated in _handle_checkpointing)
        # Use a small epsilon for floating point comparison
        if current_metric >= self.best_metric - 1e-8:
            self.early_stop_counter = 0
            return False
        else:
            self.early_stop_counter += 1
            print(f"  → Early stopping: {self.early_stop_counter}/{es_cfg['patience']} (no improvement in {metric_name})")
            return self.early_stop_counter >= es_cfg['patience']
    
    def _save_metrics_history(self):
        """Save metrics history to YAML file."""
        metrics_file = self.output_dir / 'metrics_history.yaml'
        
        # Convert metrics history to a serializable format
        serializable_history = []
        for epoch_data in self.metrics_history:
            epoch_dict = {
                'epoch': epoch_data['epoch'],
                'train': {k: float(v) for k, v in epoch_data['train'].items()},
                'val': {k: float(v) for k, v in epoch_data['val'].items()}
            }
            serializable_history.append(epoch_dict)
        
        with open(metrics_file, 'w') as f:
            yaml.dump(serializable_history, f, default_flow_style=False, sort_keys=False)
    
    def _log_sample_images(self, epoch: int):
        """
        Log sample images to TensorBoard.
        
        Args:
            epoch: Current epoch number
        """
        if self.tb_logger is None:
            return
        
        self.model.eval()
        with torch.no_grad():
            # Get one batch from validation loader
            batch = next(iter(self.val_loader))
            images = batch['image'].to(self.device) if isinstance(batch['image'], torch.Tensor) else torch.from_numpy(batch['image']).to(self.device)
            masks = batch['mask'].to(self.device) if isinstance(batch['mask'], torch.Tensor) else torch.from_numpy(batch['mask']).to(self.device)
            
            # Run inference
            outputs = self.model(images)
            
            # Log images (limit to 4 samples)
            dataset_config = self.config.get('dataset', {})
            mean = dataset_config.get('mean', [0.485, 0.456, 0.406])
            std = dataset_config.get('std', [0.229, 0.224, 0.225])
            
            self.tb_logger.log_images(
                tag='val/predictions',
                images=images,
                masks_gt=masks,
                masks_pred=outputs,
                step=epoch,
                max_images=4,
                denormalize=True,
                mean=mean,
                std=std
            )
    
    def _log_hyperparameters(self):
        """Log hyperparameters and final metrics to TensorBoard."""
        if self.tb_logger is None or len(self.metrics_history) == 0:
            return
        
        # Collect hyperparameters
        hparams = {
            'model': self.config['model']['type'],
            'batch_size': self.config['data']['batch_size'],
            'learning_rate': self.config['training']['optimizer']['learning_rate'],
            'weight_decay': self.config['training']['optimizer']['weight_decay'],
            'optimizer': self.config['training']['optimizer']['type'],
            'scheduler': self.config['training']['scheduler']['type'],
            'loss': self.config['training']['loss']['type'],
            'epochs': self.config['training']['epochs'],
        }
        
        # Get final metrics (last epoch)
        final_epoch = self.metrics_history[-1]
        final_metrics = {}
        for k, v in final_epoch['val'].items():
            # Remove 'val_' prefix for cleaner display
            clean_name = k.replace('val_', '')
            final_metrics[f'final_val_{clean_name}'] = float(v)
        
        # Add best metric
        monitor_metric = self.config['training']['checkpoint']['monitor']
        final_metrics['best_' + monitor_metric.replace('val_', '')] = float(self.best_metric)
        
        self.tb_logger.log_hyperparameters(hparams, final_metrics)
    
    def _log_activations(self, epoch: int):
        """
        Log layer activations to TensorBoard.
        
        Args:
            epoch: Current epoch number
        """
        if self.activation_logger is None or self.tb_logger is None:
            return
        
        self.model.eval()
        with torch.no_grad():
            # Get one batch from validation loader
            batch = next(iter(self.val_loader))
            images = batch['image'].to(self.device) if isinstance(batch['image'], torch.Tensor) else torch.from_numpy(batch['image']).to(self.device)
            
            # Clear previous activations
            self.activation_logger.clear_activations()
            
            # Run forward pass (activations will be captured by hooks)
            _ = self.model(images)
            
            # Log activations to TensorBoard
            activations = self.activation_logger.get_activations()
            log_activations_to_tensorboard(
                self.tb_logger,
                activations,
                epoch,
                prefix='activations'
            )
            
            # Clear activations to free memory
            self.activation_logger.clear_activations()