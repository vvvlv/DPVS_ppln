"""Dataset class for vessel segmentation."""

from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, Optional
import random


class VesselSegmentationDataset(Dataset):
    """Dataset for vessel segmentation."""
    
    def __init__(
        self,
        root_dir: str,
        split: str,
        image_size: tuple,
        mean: tuple,
        std: tuple,
        normalize: bool = True,
        num_channels: int = 3,
        augmentation: Optional[dict] = None
    ):
        """
        Initialize dataset.
        
        Args:
            root_dir: Dataset root directory
            split: "train", "val", or "test"
            image_size: Expected image size (H, W)
            mean: Normalization mean
            std: Normalization std
            normalize: Whether to normalize images
            num_channels: Number of image channels (1 for grayscale/single channel, 3 for RGB)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.normalize = normalize
        self.num_channels = num_channels
        self.augmentation = augmentation or {}
        self.use_augmentation = self.augmentation.get('enabled', False) and self.split == 'train'
        
        # Load image and mask paths
        image_dir = self.root_dir / split / "image"
        mask_dir = self.root_dir / split / "label"
        
        self.image_paths = sorted(image_dir.glob("*.png"))
        self.mask_paths = sorted(mask_dir.glob("*.png"))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Mismatch: {len(self.image_paths)} images, {len(self.mask_paths)} masks"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single sample."""
        # Load image and mask
        if self.num_channels == 1:
            # Load as grayscale
            image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        else:
            # Load as RGB
            image = cv2.imread(str(self.image_paths[idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        
        # Resize if needed
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size)
        
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        if self.use_augmentation:
            image, mask = self._apply_augmentations(image, mask)
            image = np.ascontiguousarray(image)
            mask = np.ascontiguousarray(mask)
        
        if self.normalize:
            image = (image - self.mean) / self.std
        
        # To tensor format (CHW)
        if self.num_channels == 1:
            # Add channel dimension for grayscale
            image = image[None, :, :]
        else:
            # Transpose for RGB
            image = image.transpose(2, 0, 1)
        
        mask = mask[None, :, :]
        
        return {
            "image": image,
            "mask": mask,
            "name": self.image_paths[idx].name
        } 

    def _apply_augmentations(self, image: np.ndarray, mask: np.ndarray):
        aug = self.augmentation
        # Horizontal flip
        h_flip_prob = aug.get('horizontal_flip', 0.0)
        if h_flip_prob > 0 and random.random() < h_flip_prob:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        
        # Vertical flip
        v_flip_prob = aug.get('vertical_flip', 0.0)
        if v_flip_prob > 0 and random.random() < v_flip_prob:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        
        # 90-degree rotations
        if aug.get('rotation90', False):
            rotation_prob = aug.get('rotation_prob', 0.5)
            if random.random() < rotation_prob:
                k = random.randint(1, 3)
                image = np.rot90(image, k, axes=(0, 1))
                mask = np.rot90(mask, k)
        
        # Random brightness adjustment
        brightness = aug.get('brightness', 0.0)
        if brightness > 0:
            factor = 1.0 + random.uniform(-brightness, brightness)
            image = np.clip(image * factor, 0.0, 1.0)
        
        return image, mask