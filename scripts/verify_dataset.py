#!/usr/bin/env python3
"""
Verify prepared dataset integrity.

Checks:
- Image and mask counts
- 1:1 filename matching
- Mask value ranges
- Prints diagnostic information
"""

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = {
    0: "background",
    1: "head",
    2: "hand/arm",
    3: "body/torso",
    4: "foot/leg",
}


def get_files(directory: Path, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> set:
    """Get set of filenames (without extension) in a directory."""
    files = set()
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in extensions:
            files.add(f.stem)
    return files


def verify_masks(masks_dir: Path, sample_size: int = 100) -> tuple[bool, Counter]:
    """Verify mask values and compute histogram on sample."""
    mask_files = list(masks_dir.glob('*.png'))
    
    if not mask_files:
        return False, Counter()
    
    # Sample random masks
    sample = random.sample(mask_files, min(sample_size, len(mask_files)))
    
    value_histogram = Counter()
    invalid_masks = []
    
    for mask_path in sample:
        try:
            mask = np.array(Image.open(mask_path))
            unique_values = np.unique(mask)
            
            # Check for invalid values
            for v in unique_values:
                if v not in CLASS_NAMES:
                    invalid_masks.append((mask_path.name, v))
                value_histogram[int(v)] += 1
                
        except Exception as e:
            invalid_masks.append((mask_path.name, str(e)))
    
    if invalid_masks:
        print(f"\n  WARNING: Found {len(invalid_masks)} issues in masks:")
        for name, issue in invalid_masks[:10]:
            print(f"    {name}: {issue}")
    
    return len(invalid_masks) == 0, value_histogram


def main():
    parser = argparse.ArgumentParser(description="Verify prepared dataset")
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory to verify')
    parser.add_argument('--max_missing_show', type=int, default=50,
                        help='Maximum missing files to show')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    images_dir = data_dir / 'images'
    masks_dir = data_dir / 'masks'
    splits_dir = data_dir / 'splits'
    
    errors = []
    warnings = []
    
    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    
    # Check directories exist
    print("\n[1/5] Checking directories...")
    for d in [images_dir, masks_dir, splits_dir]:
        if not d.exists():
            errors.append(f"Directory not found: {d}")
            print(f"  ERROR: {d} does not exist")
        else:
            print(f"  OK: {d}")
    
    if errors:
        print("\nCritical errors found. Exiting.")
        sys.exit(1)
    
    # Count files
    print("\n[2/5] Counting files...")
    image_files = get_files(images_dir, ('.jpg', '.jpeg', '.png'))
    mask_files = get_files(masks_dir, ('.png',))
    
    print(f"  Images: {len(image_files)}")
    print(f"  Masks: {len(mask_files)}")
    
    if len(image_files) == 0:
        errors.append("No images found")
    if len(mask_files) == 0:
        errors.append("No masks found")
    
    # Check 1:1 matching
    print("\n[3/5] Checking image-mask correspondence...")
    
    images_without_masks = image_files - mask_files
    masks_without_images = mask_files - image_files
    matched = image_files & mask_files
    
    print(f"  Matched pairs: {len(matched)}")
    print(f"  Images without masks: {len(images_without_masks)}")
    print(f"  Masks without images: {len(masks_without_images)}")
    
    if images_without_masks:
        warnings.append(f"{len(images_without_masks)} images without masks")
        print(f"\n  Missing masks for images (showing up to {args.max_missing_show}):")
        for name in sorted(images_without_masks)[:args.max_missing_show]:
            print(f"    {name}")
        if len(images_without_masks) > args.max_missing_show:
            print(f"    ... and {len(images_without_masks) - args.max_missing_show} more")
    
    if masks_without_images:
        warnings.append(f"{len(masks_without_images)} masks without images")
        print(f"\n  Masks without source images (showing up to {args.max_missing_show}):")
        for name in sorted(masks_without_images)[:args.max_missing_show]:
            print(f"    {name}")
        if len(masks_without_images) > args.max_missing_show:
            print(f"    ... and {len(masks_without_images) - args.max_missing_show} more")
    
    # Verify split files
    print("\n[4/5] Checking split files...")
    for split_name in ['train.txt', 'val.txt']:
        split_path = splits_dir / split_name
        if not split_path.exists():
            errors.append(f"Split file not found: {split_path}")
            print(f"  ERROR: {split_path} not found")
        else:
            with open(split_path, 'r') as f:
                split_files = [line.strip() for line in f if line.strip()]
            
            # Check split files exist in matched set
            missing_in_split = set(split_files) - matched
            print(f"  {split_name}: {len(split_files)} entries")
            if missing_in_split:
                warnings.append(f"{len(missing_in_split)} entries in {split_name} not in matched pairs")
                print(f"    WARNING: {len(missing_in_split)} entries reference missing pairs")
    
    # Verify mask values
    print("\n[5/5] Verifying mask values (sampling)...")
    masks_valid, value_hist = verify_masks(masks_dir)
    
    print("\n  Class value frequency in sampled masks:")
    for cls_id in sorted(CLASS_NAMES.keys()):
        count = value_hist.get(cls_id, 0)
        name = CLASS_NAMES.get(cls_id, "UNKNOWN")
        print(f"    Class {cls_id} ({name:12s}): found in {count} masks")
    
    # Check for unexpected values
    unexpected = set(value_hist.keys()) - set(CLASS_NAMES.keys())
    if unexpected:
        errors.append(f"Unexpected class values in masks: {unexpected}")
        print(f"\n  ERROR: Unexpected class values: {unexpected}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal matched pairs: {len(matched)}")
    
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    
    if not errors and not warnings:
        print("\nAll checks passed!")
    
    # Exit code
    if errors:
        print("\nVerification FAILED with critical errors.")
        sys.exit(1)
    elif warnings:
        print("\nVerification completed with warnings.")
        sys.exit(0)
    else:
        print("\nVerification PASSED.")
        sys.exit(0)


if __name__ == '__main__':
    main()
