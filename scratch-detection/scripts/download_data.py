"""
Download NEU Surface Defect Database
Run: python scripts/download_data.py
"""

import os
import zipfile
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

def main():
    print("=" * 60)
    print("NEU Surface Defect Dataset Download")
    print("=" * 60)
    
    # Create data directory
    os.makedirs('data/raw', exist_ok=True)
    
    # Option 1: Kaggle (RECOMMENDED - Easier)
    print("\n[OPTION 1] Download from Kaggle (RECOMMENDED)")
    print("1. Go to: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database")
    print("2. Click 'Download' button")
    print("3. Extract ZIP to data/raw/NEU-DET/")
    print("\nOR use Kaggle API:")
    print("   kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database")
    print("   unzip neu-surface-defect-database.zip -d data/raw/")
    
    # Option 2: Direct download (if available)
    print("\n[OPTION 2] Direct Download")
    print("Visit: http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/")
    
    print("\n" + "=" * 60)
    print("Dataset Structure Expected:")
    print("data/raw/NEU-DET/")
    print("  ├── crazing/")
    print("  ├── inclusion/")
    print("  ├── patches/")
    print("  ├── pitted_surface/")
    print("  ├── rolled-in_scale/")
    print("  └── scratches/  <-- Main focus")
    print("=" * 60)
    
    # Check if data exists
    if os.path.exists('data/raw/NEU-DET'):
        print("\n✓ Dataset found in data/raw/NEU-DET/")
        
        # Count images
        classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 
                   'rolled-in_scale', 'scratches']
        for cls in classes:
            path = f'data/raw/NEU-DET/{cls}'
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.bmp'))])
                print(f"  {cls}: {count} images")
    else:
        print("\n✗ Dataset not found. Please download manually.")
        print("  After downloading, run this script again to verify.")
    
    print("\n[NEXT STEP] Run: python scripts/prepare_data.py")

if __name__ == "__main__":
    main()