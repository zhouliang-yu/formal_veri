#!/usr/bin/env python3
"""
Download and prepare SST dataset for language model robustness experiments
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def download_file(url, output_path):
    """Download a file using wget or curl"""
    print(f"Downloading {url}...")
    
    # Try wget first, then curl
    if shutil.which('wget'):
        cmd = ['wget', url, '-O', output_path]
    elif shutil.which('curl'):
        cmd = ['curl', '-L', url, '-o', output_path]
    else:
        print("Error: Neither wget nor curl found. Please install one of them.")
        return False
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Downloaded to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e}")
        return False

def extract_tar(tar_path, extract_dir):
    """Extract tar.gz file"""
    print(f"Extracting {tar_path}...")
    try:
        subprocess.run(['tar', 'xvf', tar_path], cwd=extract_dir, check=True)
        print("Extraction completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting: {e}")
        return False

def main():
    # Get the script directory (should be in language/ directory)
    script_dir = Path(__file__).parent.absolute()
    language_dir = script_dir
    
    # Change to language directory
    os.chdir(str(language_dir))
    
    # Check if data already exists
    data_dir = language_dir / 'data' / 'sst'
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"Data directory already exists: {data_dir}")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    # Create data directory
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Download preprocessed data (easier option)
    print("\n" + "="*60)
    print("Option 1: Download preprocessed data (recommended)")
    print("="*60)
    url = "http://download.huan-zhang.com/datasets/language/data_language.tar.gz"
    tar_path = language_dir / 'data_language.tar.gz'
    
    if download_file(url, str(tar_path)):
        if extract_tar(str(tar_path), str(language_dir)):
            # Clean up tar file
            if tar_path.exists():
                tar_path.unlink()
            print("\n" + "="*60)
            print("Data download completed!")
            print("="*60)
            print(f"\nData should be in: {data_dir}")
            print("\nVerifying files...")
            required_files = ['train_all_nodes.json', 'train.json', 'dev.json', 'test.json']
            all_exist = True
            for f in required_files:
                file_path = data_dir / f
                if file_path.exists():
                    print(f"  ✓ {f}")
                else:
                    print(f"  ✗ {f} (missing)")
                    all_exist = False
            
            if all_exist:
                print("\n✓ All required files are present!")
                print("You can now run the robustness experiments.")
            else:
                print("\n⚠ Some files are missing. You may need to run preprocessing.")
        else:
            print("Failed to extract archive")
    else:
        print("\n" + "="*60)
        print("Alternative: Download raw SST data and preprocess")
        print("="*60)
        print("\nIf the preprocessed data download fails, you can:")
        print("1. Download raw SST data from:")
        print("   https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip")
        print("   https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip")
        print("\n2. Extract and place files in preprocess/ directory")
        print("3. Run: python preprocess/preprocess_sst.py")
        print("4. Move JSON files to data/sst/")

if __name__ == '__main__':
    main()