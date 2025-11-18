"""Data module."""

from torch.utils.data import DataLoader
from typing import Tuple
from .dataset import VesselSegmentationDataset


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    dataset_config = config['dataset']
    data_config = config['data']
    augmentation_config = data_config.get('augmentation', {})
    
    # Create datasets
    train_dataset = VesselSegmentationDataset(
        root_dir=dataset_config['paths']['root'],
        split='train',
        image_size=tuple(dataset_config['image_size']),
        mean=dataset_config['mean'],
        std=dataset_config['std'],
        normalize=data_config['preprocessing']['normalize'],
        num_channels=dataset_config['num_channels'],
        augmentation=augmentation_config
    )
    
    val_dataset = VesselSegmentationDataset(
        root_dir=dataset_config['paths']['root'],
        split='val',
        image_size=tuple(dataset_config['image_size']),
        mean=dataset_config['mean'],
        std=dataset_config['std'],
        normalize=data_config['preprocessing']['normalize'],
        num_channels=dataset_config['num_channels'],
        augmentation=augmentation_config
    )
    
    test_dataset = VesselSegmentationDataset(
        root_dir=dataset_config['paths']['root'],
        split='test',
        image_size=tuple(dataset_config['image_size']),
        mean=dataset_config['mean'],
        std=dataset_config['std'],
        normalize=data_config['preprocessing']['normalize'],
        num_channels=dataset_config['num_channels'],
        augmentation=augmentation_config
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader 