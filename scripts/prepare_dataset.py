#!/usr/bin/env python3
"""
Prepare dataset from JSON labels and raw frames.

Reads train_label.json and test_label.json, finds corresponding images in all_data/,
decodes base64 labels to class-id masks, and creates normalized folder structure.

Output structure:
    data/images/   - Copied input images
    data/masks/    - Single-channel class-id masks (PNG, mode L)
    data/splits/   - train.txt and val.txt with filenames
"""

import argparse
import base64
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# RGB color to class ID mapping
COLOR_TO_CLASS = {
    (0, 0, 0): 0,        # background
    (44, 160, 44): 1,    # head
    (31, 119, 180): 2,   # hand/arm
    (255, 127, 14): 3,   # body/torso
    (214, 39, 40): 4,    # foot/leg
}

CLASS_NAMES = {
    0: "background",
    1: "head",
    2: "hand/arm",
    3: "body/torso",
    4: "foot/leg",
}


def base64_to_rgb(base64_str: str) -> np.ndarray:
    """Decode base64 string to RGB numpy array."""
    base64_decoded = base64.b64decode(base64_str)
    im_arr = np.frombuffer(base64_decoded, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    # Convert BGR to RGB
    return img[:, :, ::-1]


def rgb_to_class_mask(rgb_label: np.ndarray) -> np.ndarray:
    """Convert RGB label image to single-channel class-id mask."""
    h, w = rgb_label.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.uint8)
    
    for rgb, class_id in COLOR_TO_CLASS.items():
        mask = np.all(rgb_label == rgb, axis=2)
        class_mask[mask] = class_id
    
    return class_mask


def build_image_index(all_data_dir: Path) -> dict:
    """Build index of all images in all_data/ for fast lookup."""
    index = {}  # basename -> full path
    index_lower = {}  # lowercase basename -> full path (for fallback)
    
    for root, _, files in os.walk(all_data_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = Path(root) / fname
                index[fname] = full_path
                index_lower[fname.lower()] = full_path
    
    return index, index_lower


def find_image(filename: str, index: dict, index_lower: dict) -> Path | None:
    """Find image by basename, case-sensitive first, then case-insensitive."""
    if filename in index:
        return index[filename]
    if filename.lower() in index_lower:
        return index_lower[filename.lower()]
    return None


def process_json_file(
    json_path: Path,
    all_data_index: tuple,
    images_dir: Path,
    masks_dir: Path,
    split_name: str
) -> tuple[list[str], int, Counter]:
    """Process a single JSON file and return filenames, missing count, class histogram."""
    index, index_lower = all_data_index
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    processed_files = []
    missing_count = 0
    class_histogram = Counter()
    
    for video_id, frames in data.items():
        for frame_info in frames:
            image_filename = frame_info.get('image')
            label_base64 = frame_info.get('label')
            
            if not image_filename or not label_base64:
                continue
            
            # Find source image
            src_path = find_image(image_filename, index, index_lower)
            if src_path is None:
                missing_count += 1
                continue
            
            # Get basename for output
            basename = os.path.basename(image_filename)
            basename_no_ext = os.path.splitext(basename)[0]
            
            # Copy image
            dst_image_path = images_dir / basename
            if not dst_image_path.exists():
                shutil.copy2(src_path, dst_image_path)
            
            # Decode and save mask
            try:
                rgb_label = base64_to_rgb(label_base64)
                class_mask = rgb_to_class_mask(rgb_label)
                
                # Update histogram
                unique, counts = np.unique(class_mask, return_counts=True)
                for cls_id, cnt in zip(unique, counts):
                    class_histogram[int(cls_id)] += cnt
                
                # Save mask as PNG (mode L = 8-bit grayscale)
                mask_path = masks_dir / f"{basename_no_ext}.png"
                Image.fromarray(class_mask, mode='L').save(mask_path)
                
                processed_files.append(basename_no_ext)
            except Exception as e:
                print(f"  Warning: Failed to process {basename}: {e}")
                continue
    
    return processed_files, missing_count, class_histogram


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from JSON labels")
    parser.add_argument('--all_data_dir', type=str, default='all_data',
                        help='Directory containing raw frames')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for processed data')
    parser.add_argument('--train_json', type=str, default='train_label.json',
                        help='Path to training labels JSON')
    parser.add_argument('--test_json', type=str, default='test_label.json',
                        help='Path to test labels JSON')
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path('.')
    all_data_dir = base_dir / args.all_data_dir
    output_dir = base_dir / args.output_dir
    
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    splits_dir = output_dir / 'splits'
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Building image index from {all_data_dir}...")
    image_index = build_image_index(all_data_dir)
    print(f"  Found {len(image_index[0])} images")
    
    total_histogram = Counter()
    
    # Process training data
    print(f"\nProcessing {args.train_json}...")
    train_files, train_missing, train_hist = process_json_file(
        base_dir / args.train_json,
        image_index,
        images_dir,
        masks_dir,
        'train'
    )
    total_histogram.update(train_hist)
    print(f"  Processed: {len(train_files)} samples")
    print(f"  Missing images: {train_missing}")
    
    # Process test/val data
    print(f"\nProcessing {args.test_json}...")
    val_files, val_missing, val_hist = process_json_file(
        base_dir / args.test_json,
        image_index,
        images_dir,
        masks_dir,
        'val'
    )
    total_histogram.update(val_hist)
    print(f"  Processed: {len(val_files)} samples")
    print(f"  Missing images: {val_missing}")
    
    # Write split files
    train_split_path = splits_dir / 'train.txt'
    val_split_path = splits_dir / 'val.txt'
    
    with open(train_split_path, 'w') as f:
        f.write('\n'.join(sorted(set(train_files))))
    
    with open(val_split_path, 'w') as f:
        f.write('\n'.join(sorted(set(val_files))))
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total training samples: {len(train_files)}")
    print(f"Total validation samples: {len(val_files)}")
    print(f"Total missing images: {train_missing + val_missing}")
    print(f"\nOutput directories:")
    print(f"  Images: {images_dir}")
    print(f"  Masks: {masks_dir}")
    print(f"  Splits: {splits_dir}")
    
    print(f"\nClass pixel histogram:")
    for cls_id in sorted(CLASS_NAMES.keys()):
        count = total_histogram.get(cls_id, 0)
        pct = 100 * count / sum(total_histogram.values()) if total_histogram else 0
        print(f"  {cls_id} ({CLASS_NAMES[cls_id]:12s}): {count:>12,} pixels ({pct:5.2f}%)")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
