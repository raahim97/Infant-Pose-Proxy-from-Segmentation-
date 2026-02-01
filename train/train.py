#!/usr/bin/env python3
"""
Training script for infant body part segmentation.

Fine-tunes DeepLabV3 with MobileNetV3 backbone on the prepared dataset.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from dataset import get_dataloaders


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict:
    """Compute IoU per class and mean IoU."""
    ious = {}
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        
        if union > 0:
            ious[cls] = intersection / union
        else:
            ious[cls] = float('nan')
    
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    ious['mean'] = np.mean(valid_ious) if valid_ious else 0.0
    
    return ious


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    num_classes: int
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        # Collect predictions for IoU
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)}, Loss: {loss.item():.4f}")
    
    # Compute epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    ious = compute_iou(all_preds, all_targets, num_classes)
    
    return {
        'loss': total_loss / total_samples,
        'miou': ious['mean'],
        'class_ious': ious
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int
) -> dict:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    ious = compute_iou(all_preds, all_targets, num_classes)
    
    return {
        'loss': total_loss / total_samples,
        'miou': ious['mean'],
        'class_ious': ious
    }


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create DeepLabV3 model with MobileNetV3 backbone."""
    model = deeplabv3_mobilenet_v3_large(
        weights='DEFAULT' if pretrained else None,
        weights_backbone='DEFAULT' if pretrained else None
    )
    
    # Replace classifier head for our number of classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    # Also update auxiliary classifier if it exists
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(40, num_classes, kernel_size=1)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train infant segmentation model")
    parser.add_argument('--images_dir', type=str, default='data/images',
                        help='Directory with input images')
    parser.add_argument('--masks_dir', type=str, default='data/masks',
                        help='Directory with mask images')
    parser.add_argument('--train_split', type=str, default='data/splits/train.txt',
                        help='Training split file')
    parser.add_argument('--val_split', type=str, default='data/splits/val.txt',
                        help='Validation split file')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of segmentation classes')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--out', type=str, default='models/segmentation.pt',
                        help='Output checkpoint path')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda, mps)')
    args = parser.parse_args()
    
    # Select device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader = get_dataloaders(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(num_classes=args.num_classes, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    best_miou = 0.0
    class_names = ['background', 'head', 'hand/arm', 'body/torso', 'foot/leg']
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.num_classes
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, args.num_classes
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics
        epoch_time = time.time() - start_time
        print(f"\n  Train Loss: {train_metrics['loss']:.4f}, mIoU: {train_metrics['miou']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, mIoU: {val_metrics['miou']:.4f}")
        print(f"  Time: {epoch_time:.1f}s, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Print per-class IoU
        print("  Val IoU per class:")
        for cls_id, cls_name in enumerate(class_names):
            iou = val_metrics['class_ious'].get(cls_id, float('nan'))
            if not np.isnan(iou):
                print(f"    {cls_name}: {iou:.4f}")
        
        # Save best model
        if val_metrics['miou'] > best_miou:
            best_miou = val_metrics['miou']
            print(f"\n  New best mIoU: {best_miou:.4f} - Saving checkpoint...")
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'num_classes': args.num_classes,
                'image_size': args.image_size,
                'class_names': class_names
            }
            torch.save(checkpoint, args.out)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Checkpoint saved to: {args.out}")


if __name__ == '__main__':
    main()
