"""TensorBoard logging utilities."""

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np


class TensorBoardLogger:
    """
    TensorBoard logger for training visualization.
    
    Logs scalars, images, and model graphs to TensorBoard.
    Conditionally enabled based on configuration.
    """
    
    def __init__(self, log_dir: str, enabled: bool = True):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.writer: Optional[SummaryWriter] = None
        
        if self.enabled:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_dir))
            print(f"TensorBoard logging enabled: {log_dir}")
            print(f"  View logs with: tensorboard --logdir {log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value.
        
        Args:
            tag: Name of the scalar (e.g., 'train/loss')
            value: Scalar value
            step: Global step (epoch or iteration)
        """
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """
        Log multiple scalars under a main tag.
        
        Args:
            main_tag: Main category (e.g., 'metrics')
            tag_scalar_dict: Dictionary of {tag: value}
            step: Global step
        """
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str, step: int):
        """
        Log all metrics from a dictionary.
        
        Args:
            metrics: Dictionary of metric names and values
            prefix: Prefix for tag names (e.g., 'train' or 'val')
            step: Global step (epoch number)
        """
        if self.enabled and self.writer is not None:
            for name, value in metrics.items():
                # Remove 'val_' prefix if it exists to avoid double prefix
                clean_name = name.replace('val_', '')
                tag = f"{prefix}/{clean_name}"
                self.writer.add_scalar(tag, value, step)
    
    def log_learning_rate(self, lr: float, step: int):
        """
        Log current learning rate.
        
        Args:
            lr: Learning rate value
            step: Global step (epoch number)
        """
        if self.enabled and self.writer is not None:
            self.writer.add_scalar('learning_rate', lr, step)
    
    def log_images(
        self,
        tag: str,
        images: torch.Tensor,
        masks_gt: torch.Tensor,
        masks_pred: torch.Tensor,
        step: int,
        max_images: int = 4,
        denormalize: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ):
        """
        Log image comparisons (input, ground truth, prediction).
        
        Args:
            tag: Tag for the image grid (e.g., 'val/samples')
            images: Input images (B, C, H, W)
            masks_gt: Ground truth masks (B, 1, H, W)
            masks_pred: Predicted masks (B, 1, H, W)
            step: Global step (epoch number)
            max_images: Maximum number of images to log
            denormalize: Whether to denormalize images
            mean: Mean for denormalization
            std: Std for denormalization
        """
        if not self.enabled or self.writer is None:
            return
        
        # Limit number of images
        n = min(images.size(0), max_images)
        images = images[:n]
        masks_gt = masks_gt[:n]
        masks_pred = masks_pred[:n]
        
        # Denormalize images if needed
        if denormalize:
            num_channels = images.size(1)
            mean_tensor = torch.tensor(mean, device=images.device).view(1, num_channels, 1, 1)
            std_tensor = torch.tensor(std, device=images.device).view(1, num_channels, 1, 1)
            images = images * std_tensor + mean_tensor
            images = torch.clamp(images, 0, 1)
        
        # Convert grayscale images to RGB for visualization if needed
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Convert grayscale masks to RGB for visualization
        masks_gt_rgb = masks_gt.repeat(1, 3, 1, 1)
        masks_pred = (masks_pred > 0.5).float() 
        masks_pred_rgb = masks_pred.repeat(1, 3, 1, 1)

        # Colors: True-Positive = Green, False-Positive = Red, False-Negative = Blue
        true_positive = (masks_gt * masks_pred)                     
        false_positive = ((1.0 - masks_gt) * masks_pred)           
        false_negative = (masks_gt * (1.0 - masks_pred))  

        overlay_rgb = torch.zeros_like(masks_pred_rgb)   
        # FN
        overlay_rgb[:, 0, :, :] = false_positive[:, 0, :, :] + true_positive[:, 0, :, :] + false_negative[:, 0, :, :]     
        # FP
        overlay_rgb[:, 1, :, :] = false_positive[:, 0, :, :] + true_positive[:, 0, :, :]    
        # TP    
        overlay_rgb[:, 2, :, :] = true_positive[:, 0, :, :]  

        # Show on real image
        # transparency of overlay
        alpha = 0.5  
        overlay_on_img = torch.clamp(images * (1.0 - alpha) + overlay_rgb * alpha, 0.0, 1.0)
        
        # Create grid: [image, ground truth, prediction, overlay, overlay on img] for each sample
        comparison_list = []
        for i in range(n):
            comparison_list.extend([images[i], masks_gt_rgb[i], masks_pred_rgb[i], overlay_rgb[i], overlay_on_img[i]])
        
        # Stack and create grid (nrow=5 shows [image, gt, pred, overlay, overlay on img] per row)
        comparison = torch.stack(comparison_list)
        grid = vutils.make_grid(comparison, nrow=5, padding=2, normalize=False)
        
        self.writer.add_image(tag, grid, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_size: tuple, device: str = 'cuda'):
        """
        Log model architecture graph.
        
        Args:
            model: PyTorch model
            input_size: Input tensor size (B, C, H, W)
            device: Device to create dummy input on
        """
        if not self.enabled or self.writer is None:
            return
        
        try:
            dummy_input = torch.randn(input_size).to(device)
            self.writer.add_graph(model, dummy_input)
            self.writer.flush()
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
    
    def log_hyperparameters(self, hparam_dict: Dict, metric_dict: Dict):
        """
        Log hyperparameters and their resulting metrics.
        
        Args:
            hparam_dict: Dictionary of hyperparameters
            metric_dict: Dictionary of resulting metrics
        """
        if self.enabled and self.writer is not None:
            self.writer.add_hparams(hparam_dict, metric_dict)
    
    def log_text(self, tag: str, text: str, step: int = 0):
        """
        Log text information.
        
        Args:
            tag: Tag for the text
            text: Text content
            step: Global step
        """
        if self.enabled and self.writer is not None:
            self.writer.add_text(tag, text, step)
    
    def flush(self):
        """Flush pending events to disk."""
        if self.enabled and self.writer is not None:
            self.writer.flush()
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()
            print("TensorBoard logger closed.")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()


class ActivationLogger:
    """
    Logger for capturing and logging layer activations to TensorBoard.
    
    Supports registering forward hooks on specific layers to capture
    their activations during forward pass, then logs histograms and
    statistics to TensorBoard.
    """
    
    def __init__(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        """
        Initialize activation logger.
        
        Args:
            model: PyTorch model to monitor
            layer_names: List of layer names to monitor (e.g., ['encoder1', 'bottleneck'])
                        If None, monitors all named modules
        """
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self.layer_names = layer_names
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        if self.layer_names is None:
            # Monitor all named modules
            for name, module in self.model.named_modules():
                if len(name) > 0:  # Skip root module
                    hook = module.register_forward_hook(
                        self._create_hook(name)
                    )
                    self.hooks.append(hook)
        else:
            # Monitor specific layers
            module_dict = dict(self.model.named_modules())
            for name in self.layer_names:
                if name in module_dict:
                    hook = module_dict[name].register_forward_hook(
                        self._create_hook(name)
                    )
                    self.hooks.append(hook)
                else:
                    print(f"Warning: Layer '{name}' not found in model")
    
    def _create_hook(self, name: str):
        """Create a forward hook that captures activations."""
        def hook(module, input, output):
            # Store activation (detach to avoid keeping computation graph)
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach()
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                # For modules that return tuples, take first element
                if isinstance(output[0], torch.Tensor):
                    self.activations[name] = output[0].detach()
        return hook
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get captured activations."""
        return self.activations
    
    def clear_activations(self):
        """Clear stored activations to free memory."""
        self.activations.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self.remove_hooks()


def get_default_layer_names(model_type: str) -> List[str]:
    """
    Get default layer names to monitor based on model type.
    
    Args:
        model_type: Model architecture name (e.g., 'UNet', 'RoiNet')
    
    Returns:
        List of layer names to monitor
    """
    # Define sensible defaults for each model type
    defaults = {
        'UNet': [
            'down_conv_1', 'down_conv_2', 'down_conv_3', 'down_conv_4',
            'bottleneck',
            'up_conv_1', 'up_conv_2', 'up_conv_3', 'up_conv_4'
        ],
        'RoiNet': [
            'dict_module.conv0',
            'dict_module.conv1',
            'dict_module.conv2',
            'dict_module.bottle1',
            'dict_module.bottle2',
            'dict_module.conv3',
            'dict_module.conv4',
            'dict_module.conv5'
        ],
        'UTrans': [
            'encoder1', 'encoder2', 'encoder3', 'encoder4',
            'bottleneck_conv_in',
            'transformer',
            'bottleneck_conv_out',
            'decoder1', 'decoder2', 'decoder3', 'decoder4'
        ],
        'TransRoiNet': [
            'dict_module.conv0',
            'dict_module.conv1',
            'dict_module.conv2',
            'dict_module.bottle1',
            'dict_module.transformer',
            'dict_module.bottle2',
            'dict_module.conv3',
            'dict_module.conv4',
            'dict_module.conv5'
        ],
        'TinySwinUNet': [
            # patch embedding
            'patch_embed',

            # first block of each encoder stage
            'encoder_layers.0.0',
            'encoder_layers.1.0',
            'encoder_layers.2.0',
            'encoder_layers.3.0',

            # patch merging layers (downsampling between stages)
            'patch_merging_layers.0',
            'patch_merging_layers.1',
            'patch_merging_layers.2',

            # upsampling stages
            'upsampling_layers.0',
            'upsampling_layers.1',
            'upsampling_layers.2',

            # first block of each decoder stage
            'decoder_layers.0.0',
            'decoder_layers.1.0',
            'decoder_layers.2.0',
        ]
    }
    
    return defaults.get(model_type, [])


def log_activations_to_tensorboard(
    tb_logger,
    activations: Dict[str, torch.Tensor],
    epoch: int,
    prefix: str = 'activations'
):
    """
    Log activation statistics to TensorBoard.
    
    Args:
        tb_logger: TensorBoard logger instance
        activations: Dictionary of {layer_name: activation_tensor}
        epoch: Current epoch number
        prefix: Prefix for TensorBoard tags
    """
    if tb_logger is None or not tb_logger.enabled:
        return
    
    for layer_name, activation in activations.items():
        # Clean layer name for TensorBoard (replace dots with slashes)
        clean_name = layer_name.replace('.', '/')
        tag = f'{prefix}/{clean_name}'
        
        # Log histogram of activations
        tb_logger.writer.add_histogram(
            tag + '/histogram',
            activation.flatten(),
            epoch
        )
        
        # Log activation statistics as scalars
        stats = {
            'mean': activation.mean().item(),
            'std': activation.std().item(),
            'min': activation.min().item(),
            'max': activation.max().item()
        }
        
        for stat_name, stat_value in stats.items():
            tb_logger.writer.add_scalar(
                f'{tag}/{stat_name}',
                stat_value,
                epoch
            )


