"""
Data loading and augmentation for scratch detection
"""

import json
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ScratchDataset(Dataset):
    """Binary scratch detection dataset"""
    
    def __init__(self, split='train', data_dir='data/processed', 
                 split_file='data/splits/splits.json', transform=None):
        """
        Args:
            split: 'train', 'val', or 'test'
            data_dir: Directory with processed images
            split_file: JSON file with splits
            transform: Albumentations transform
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load split data
        with open(split_file, 'r') as f:
            splits = json.load(f)
        
        self.data = splits[split]
        
        print(f"Loaded {split} split: {len(self.data)} images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        img_path = self.data_dir / item['path']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = item['label']
        
        return image, label

def get_train_transforms(img_size=224):
    """Heavy augmentation for training"""
    return A.Compose([
        # Resize
        A.Resize(img_size, img_size),
        
        # Geometric transforms (scratches can be any orientation)
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=180, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, 
                           rotate_limit=45, p=0.7),
        
        # Elastic deformation (simulate material warping)
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        
        # Optical distortion
        A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.3),
        
        # Color/Intensity transforms (lighting variations)
        A.CLAHE(clip_limit=2.0, p=0.5),  # Contrast enhancement
        A.RandomBrightnessContrast(brightness_limit=0.3, 
                                   contrast_limit=0.3, p=0.7),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, 
                             val_shift_limit=20, p=0.3),
        
        # Noise (sensor variations)
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ISONoise(p=0.2),
        
        # Blur (out-of-focus)
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        # Occlusions (prevent overfitting to full scratches)
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, 
                        min_holes=1, min_height=8, min_width=8, 
                        fill_value=0, p=0.3),
        
        # Normalize to ImageNet stats
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        
        ToTensorV2(),
    ])

def get_val_transforms(img_size=224):
    """Minimal transforms for validation"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def get_test_transforms(img_size=224):
    """Test-time augmentation (TTA)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def get_dataloaders(batch_size=32, num_workers=4, img_size=224):
    """Create train/val/test dataloaders"""
    
    # Create datasets
    train_dataset = ScratchDataset(
        split='train',
        transform=get_train_transforms(img_size)
    )
    
    val_dataset = ScratchDataset(
        split='val',
        transform=get_val_transforms(img_size)
    )
    
    test_dataset = ScratchDataset(
        split='test',
        transform=get_test_transforms(img_size)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)
    
    # Get one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Label distribution: Scratch={labels.sum().item()}, No-Scratch={len(labels)-labels.sum().item()}")
    print("✓ Data loader working correctly!")