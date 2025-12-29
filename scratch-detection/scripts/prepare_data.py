"""
Prepare NEU dataset for binary classification (Scratch vs No-Scratch)
Updated for train/validation/images/CLASS_NAME/ structure
Run: python scripts/prepare_data.py
"""

import os
import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

def prepare_binary_classification():
    """Convert NEU dataset to binary classification"""
    
    print("=" * 60)
    print("Preparing Binary Classification Dataset")
    print("=" * 60)
    
    # Define paths
    source_base = Path('data/raw/NEU-DET')
    target_dir = Path('data/processed')
    
    # Create target directories
    (target_dir / 'scratch').mkdir(parents=True, exist_ok=True)
    (target_dir / 'no_scratch').mkdir(parents=True, exist_ok=True)
    
    # Define class mappings
    scratch_classes = ['scratches']
    no_scratch_classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale']
    
    print("\n[1/3] Collecting and copying images...")
    
    scratch_count = 0
    no_scratch_count = 0
    
    # Process TRAIN directory
    train_images_dir = source_base / 'train' / 'images'
    if train_images_dir.exists():
        print(f"\n  Processing TRAIN directory...")
        
        # Copy scratch images
        for scratch_class in scratch_classes:
            class_dir = train_images_dir / scratch_class
            if class_dir.exists():
                for img in class_dir.glob('*.jpg'):
                    dest = target_dir / 'scratch' / f"train_{img.name}"
                    shutil.copy(img, dest)
                    scratch_count += 1
                print(f"    ✓ {scratch_class}: {len(list(class_dir.glob('*.jpg')))} images")
        
        # Copy no-scratch images
        for no_scratch_class in no_scratch_classes:
            class_dir = train_images_dir / no_scratch_class
            if class_dir.exists():
                class_count = 0
                for img in class_dir.glob('*.jpg'):
                    dest = target_dir / 'no_scratch' / f"train_{img.name}"
                    shutil.copy(img, dest)
                    class_count += 1
                    no_scratch_count += 1
                print(f"    ✓ {no_scratch_class}: {class_count} images")
    
    # Process VALIDATION directory
    val_images_dir = source_base / 'validation' / 'images'
    if val_images_dir.exists():
        print(f"\n  Processing VALIDATION directory...")
        
        # Copy scratch images
        for scratch_class in scratch_classes:
            class_dir = val_images_dir / scratch_class
            if class_dir.exists():
                for img in class_dir.glob('*.jpg'):
                    dest = target_dir / 'scratch' / f"val_{img.name}"
                    shutil.copy(img, dest)
                    scratch_count += 1
                print(f"    ✓ {scratch_class}: {len(list(class_dir.glob('*.jpg')))} images")
        
        # Copy no-scratch images
        for no_scratch_class in no_scratch_classes:
            class_dir = val_images_dir / no_scratch_class
            if class_dir.exists():
                class_count = 0
                for img in class_dir.glob('*.jpg'):
                    dest = target_dir / 'no_scratch' / f"val_{img.name}"
                    shutil.copy(img, dest)
                    class_count += 1
                    no_scratch_count += 1
                print(f"    ✓ {no_scratch_class}: {class_count} images")
    
    print(f"\n{'='*60}")
    print(f"TOTAL IMAGES COPIED:")
    print(f"  ✓ SCRATCH images: {scratch_count}")
    print(f"  ✓ NO-SCRATCH images: {no_scratch_count}")
    print(f"  ✓ TOTAL: {scratch_count + no_scratch_count}")
    print(f"{'='*60}")
    
    if scratch_count == 0 or no_scratch_count == 0:
        print("\n✗ ERROR: Missing images!")
        return
    
    # Create train/val/test splits
    create_splits(target_dir)

def create_splits(data_dir):
    """Create stratified train/val/test splits"""
    
    print("\n[2/3] Creating Train/Val/Test Splits...")
    
    # Get all images
    scratch_imgs = list((data_dir / 'scratch').glob('*'))
    no_scratch_imgs = list((data_dir / 'no_scratch').glob('*'))
    
    # Create labels
    data = []
    for img in scratch_imgs:
        data.append({'path': str(img.relative_to(data_dir)), 'label': 1})
    for img in no_scratch_imgs:
        data.append({'path': str(img.relative_to(data_dir)), 'label': 0})
    
    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    # Split: 70% train, 15% val, 15% test
    labels = [d['label'] for d in data]
    
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, test_size=0.3, stratify=labels, random_state=42
    )
    
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # Save splits
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    split_file = Path('data/splits/splits.json')
    split_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(split_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    # Print statistics
    print(f"\n  ✓ Train: {len(train_data)} images")
    print(f"      - Scratch: {sum(train_labels)}")
    print(f"      - No-Scratch: {len(train_labels) - sum(train_labels)}")
    
    print(f"\n  ✓ Val: {len(val_data)} images")
    print(f"      - Scratch: {sum(val_labels)}")
    print(f"      - No-Scratch: {len(val_labels) - sum(val_labels)}")
    
    print(f"\n  ✓ Test: {len(test_data)} images")
    print(f"      - Scratch: {sum(test_labels)}")
    print(f"      - No-Scratch: {len(test_labels) - sum(test_labels)}")
    
    print(f"\n  ✓ Splits saved to {split_file}")
    print(f"\n{'='*60}")
    print("✅ DATA PREPARATION COMPLETE!")
    print(f"{'='*60}")
    print("\n[NEXT STEP] Run: python src/train.py")

if __name__ == "__main__":
    prepare_binary_classification()