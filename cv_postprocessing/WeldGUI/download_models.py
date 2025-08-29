#!/usr/bin/env python3
"""
Direct download of PaddleOCR models without initializing PaddleOCR
"""
import os
import requests
import tarfile
import sys
from pathlib import Path

def download_file(url, destination):
    """Download a file with progress tracking"""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
    
    print(f"\nDownloaded to: {destination}")
    return destination

def extract_tar(tar_path, extract_to=None):
    """Extract tar file"""
    if extract_to is None:
        extract_to = os.path.dirname(tar_path)
    
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_to)
    
    # Remove the tar file to save space
    os.remove(tar_path)
    print(f"Extracted and removed {tar_path}")

def main():
    try:
        # Create model directory with proper structure
        model_base = Path('/opt/render/.paddleocr/whl')
        
        # Model URLs - using correct structure for PaddleOCR
        models = {
            'det': {
                'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar',
                'dir': model_base / 'det' / 'en' / 'en_PP-OCRv3_det_infer'
            },
            'rec': {
                'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar',
                'dir': model_base / 'rec' / 'en' / 'en_PP-OCRv4_rec_infer'
            },
            'cls': {
                'url': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
                'dir': model_base / 'cls' / 'ch_ppocr_mobile_v2.0_cls_infer'
            }
        }
        
        # Download each model
        for model_type, model_info in models.items():
            print(f"\n=== Downloading {model_type} model ===")
            
            # Create model-specific directory
            model_info['dir'].mkdir(exist_ok=True, parents=True)
            
            # Download the tar file to a temporary location
            tar_filename = model_info['url'].split('/')[-1]
            temp_dir = model_info['dir'].parent
            tar_path = temp_dir / tar_filename
            
            download_file(model_info['url'], str(tar_path))
            extract_tar(str(tar_path), str(temp_dir))
        
        print("\n✅ All models downloaded and extracted successfully!")
        
        # Verify the files exist
        print("\n=== Verifying downloaded files ===")
        required_files = {
            'det': ['inference.pdmodel', 'inference.pdiparams'],
            'rec': ['inference.pdmodel', 'inference.pdiparams'],
            'cls': ['inference.pdmodel', 'inference.pdiparams']
        }
        
        all_good = True
        for model_type, files in required_files.items():
            model_dir = models[model_type]['dir']
            for file in files:
                full_path = model_dir / file
                if full_path.exists():
                    print(f"✅ Found: {full_path}")
                else:
                    print(f"❌ Missing: {full_path}")
                    all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)