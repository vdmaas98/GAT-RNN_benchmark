"""
Helper script to download and prepare METR-LA dataset.

Note: This script provides instructions and basic setup.
You'll need to manually download the dataset from the official source.
"""

import os
import sys


def setup_data_directory():
    """Create data directory structure"""
    data_dir = './data/METR-LA'
    os.makedirs(data_dir, exist_ok=True)
    print(f"Created directory: {data_dir}")
    return data_dir


def check_data_files(data_dir):
    """Check if required data files exist"""
    required_files = [
        'metr-la.h5',
        'adj_mx.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
        else:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✓ Found: {file} ({file_size:.2f} MB)")
    
    return missing_files


def print_download_instructions():
    """Print instructions for downloading METR-LA dataset"""
    print("\n" + "="*70)
    print("METR-LA Dataset Download Instructions")
    print("="*70)
    print("\nOption 1: Google Drive (Recommended)")
    print("-" * 70)
    print("1. Visit: https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX")
    print("2. Download the following files:")
    print("   - metr-la.h5")
    print("   - adj_mx.pkl")
    print("3. Place them in: ./data/METR-LA/")
    
    print("\nOption 2: Original DCRNN Repository")
    print("-" * 70)
    print("1. Visit: https://github.com/liyaguang/DCRNN")
    print("2. Follow their data download instructions")
    print("3. Copy metr-la.h5 and adj_mx.pkl to: ./data/METR-LA/")
    
    print("\nOption 3: Command Line (if you have gdown)")
    print("-" * 70)
    print("pip install gdown")
    print("# Then download files using their Google Drive IDs")
    
    print("\n" + "="*70)
    print("After downloading, run this script again to verify.")
    print("="*70 + "\n")


def main():
    print("METR-LA Dataset Setup\n")
    
    data_dir = setup_data_directory()
    
    missing_files = check_data_files(data_dir)
    
    if missing_files:
        print(f"\n⚠ Missing files: {', '.join(missing_files)}")
        print_download_instructions()
        sys.exit(1)
    else:
        print("\n✓ All required files found!")
        print("\nYou can now run training:")
        print("  python train.py --model gatlstm --data_dir ./data/METR-LA --gpu 0")
        sys.exit(0)


if __name__ == '__main__':
    main()
