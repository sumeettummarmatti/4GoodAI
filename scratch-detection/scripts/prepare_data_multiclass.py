"""
Prepare NEU dataset for MULTI-CLASS classification (6 defect types)
Run: python scripts/prepare_data_multiclass.py
"""

import os
import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

def prepare_multiclass():
    """Convert NEU dataset to 6-class classification"""
    
    print("=" * 60)
    print("MULTI-CLASS CLASSIFICATION (6 Defect Types)")
    print("=" * 60)
    
    # Define paths
    source_base = Path('data/raw/NEU-DET')
    target_dir = Path('data/processed')
    
    # All 6 classes
    all_classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    
    # Create target directories for each class
    for cls in all_classes:
        (target_dir / cls).mkdir(parents=True, exist_ok=True)
    
    print("\n[1/3] Collecting and copying images...")
    
    class_counts = {cls: 0 for cls in all_classes}
    
    # Process TRAIN directory
    train_images_dir = source_base / 'train' / 'images'
    if train_images_dir.exists():
        print(f"\n  Processing TRAIN directory...")
        
        for cls in all_classes:
            class_dir = train_images_dir / cls
            if class_dir.exists():
                for img in class_dir.glob('*.jpg'):
                    dest = target_dir / cls / f"train_{img.name}"
                    shutil.copy(img, dest)
                    class_counts[cls] += 1
                print(f"    ✓ {cls}: {len(list(class_dir.glob('*.jpg')))} images")
    
    # Process VALIDATION directory
    val_images_dir = source_base / 'validation' / 'images'
    if val_images_dir.exists():
        print(f"\n  Processing VALIDATION directory...")
        
        for cls in all_classes:
            class_dir = val_images_dir / cls
            if class_dir.exists():
                for img in class_dir.glob('*.jpg'):
                    dest = target_dir / cls / f"val_{img.name}"
                    shutil.copy(img, dest)
                    class_counts[cls] += 1
                print(f"    ✓ {cls}: {len(list(class_dir.glob('*.jpg')))} images")
    
    print(f"\n{'='*60}")
    print(f"TOTAL IMAGES COPIED:")
    for cls, count in class_counts.items():
        print(f"  ✓ {cls}: {count} images")
    print(f"  ✓ TOTAL: {sum(class_counts.values())}")
    print(f"{'='*60}")
    
    # Create train/val/test splits
    create_splits(target_dir, all_classes)

def create_splits(data_dir, all_classes):
    """Create stratified train/val/test splits"""
    
    print("\n[2/3] Creating Train/Val/Test Splits...")
    
    # Collect all images with labels
    data = []
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    
    for cls in all_classes:
        class_dir = data_dir / cls
        if class_dir.exists():
            for img in class_dir.glob('*'):
                data.append({
                    'path': str(img.relative_to(data_dir)),
                    'label': class_to_idx[cls],
                    'class_name': cls
                })
    
    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    # Extract labels for stratification
    labels = [d['label'] for d in data]
    
    # Split: 70% train, 15% val, 15% test
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
        'test': test_data,
        'class_to_idx': class_to_idx,
        'idx_to_class': {idx: cls for cls, idx in class_to_idx.items()}
    }
    
    split_file = Path('data/splits/splits.json')
    split_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(split_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    # Print statistics
    print(f"\n  ✓ Train: {len(train_data)} images")
    for cls_idx in range(len(all_classes)):
        count = sum(1 for l in train_labels if l == cls_idx)
        print(f"      - {all_classes[cls_idx]}: {count}")
    
    print(f"\n  ✓ Val: {len(val_data)} images")
    for cls_idx in range(len(all_classes)):
        count = sum(1 for l in val_labels if l == cls_idx)
        print(f"      - {all_classes[cls_idx]}: {count}")
    
    print(f"\n  ✓ Test: {len(test_data)} images")
    for cls_idx in range(len(all_classes)):
        count = sum(1 for l in test_labels if l == cls_idx)
        print(f"      - {all_classes[cls_idx]}: {count}")
    
    print(f"\n  ✓ Splits saved to {split_file}")
    print(f"\n{'='*60}")
    print("✅ MULTI-CLASS DATA PREPARATION COMPLETE!")
    print(f"{'='*60}")
    print("\n[NEXT STEP] Run: python src/train.py")

if __name__ == "__main__":
    prepare_multiclass()