#!/usr/bin/env python3
"""
PyTorch Dataset for infant body part segmentation.

Reads preprocessed images and masks from data/ directory.
"""

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class InfantSegmentationDataset(Dataset):
    """
    Dataset for infant body part segmentation.
    
    Reads from:
        images_dir/   - Input RGB images
        masks_dir/    - Single-channel class ID masks (PNG, mode L)
        split_file    - Text file with basenames (one per line)
    
    Returns:
        image: float32 tensor CHW in [0, 1]
        mask: int64 tensor HW with class IDs
    """
    
    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        split_file: str | Path,
        image_size: int = 256,
        augment: bool = False,
        num_classes: int = 5
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.augment = augment
        self.num_classes = num_classes
        
        # Load split file
        with open(split_file, 'r') as f:
            self.samples = [line.strip() for line in f if line.strip()]
        
        # Filter to samples that exist
        valid_samples = []
        for basename in self.samples:
            # Try common image extensions
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = self.images_dir / f"{basename}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            
            mask_path = self.masks_dir / f"{basename}.png"
            
            if img_path is not None and mask_path.exists():
                valid_samples.append((basename, img_path, mask_path))
        
        self.samples = valid_samples
        print(f"Dataset: {len(self.samples)} valid samples from {split_file}")
        
        # Transforms
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        basename, img_path, mask_path = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask = Image.open(mask_path)
        
        # Resize both to same size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Convert to numpy
        image = np.array(image, dtype=np.float32) / 255.0  # HWC, [0, 1]
        mask = np.array(mask, dtype=np.int64)  # HW
        
        # Data augmentation (if enabled)
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            
            # Random brightness/contrast
            if np.random.random() > 0.5:
                brightness = 0.8 + 0.4 * np.random.random()
                image = np.clip(image * brightness, 0, 1)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # CHW
        mask = torch.from_numpy(mask)  # HW
        
        # Normalize image for pretrained model
        image = self.normalize(image)
        
        return image, mask


def get_dataloaders(
    images_dir: str,
    masks_dir: str,
    train_split: str,
    val_split: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
    num_classes: int = 5
) -> tuple:
    """Create training and validation dataloaders."""
    
    train_dataset = InfantSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        split_file=train_split,
        image_size=image_size,
        augment=True,
        num_classes=num_classes
    )
    
    val_dataset = InfantSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        split_file=val_split,
        image_size=image_size,
        augment=False,
        num_classes=num_classes
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Quick test
    dataset = InfantSegmentationDataset(
        images_dir='data/images',
        masks_dir='data/masks',
        split_file='data/splits/train.txt',
        image_size=256,
        augment=False
    )
    
    if len(dataset) > 0:
        image, mask = dataset[0]
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Mask unique values: {torch.unique(mask).tolist()}")
    else:
        print("No samples found. Run prepare_dataset.py first.")
