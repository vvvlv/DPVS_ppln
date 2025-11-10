"""Dataset class for vessel segmentation."""

from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
from typing import Dict


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
        num_channels: int = 3
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
        
        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0
        if self.normalize:
            image = (image - self.mean) / self.std
        
        # To tensor format (CHW)
        if self.num_channels == 1:
            # Add channel dimension for grayscale
            image = image[None, :, :]
        else:
            # Transpose for RGB
            image = image.transpose(2, 0, 1)
        
        mask = mask[None, :, :].astype(np.float32) / 255.0
        
        return {
            "image": image,
            "mask": mask,
            "name": self.image_paths[idx].name
        } 