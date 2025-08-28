#!/usr/bin/env python3
"""
Script to download PaddleOCR models during build phase
"""
import os
import sys
from paddleocr import PaddleOCR
import argparse

def download_models():
    print("Starting model download during build phase...")
    
    # Set the download path (same as runtime)
    model_path = '/opt/render/.paddleocr'
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # Download models with minimal settings
        print("Downloading detection model...")
        ocr = PaddleOCR(
            lang="en",
            use_angle_cls=False,
            use_gpu=False,
            show_log=False,
            rec_model_dir=model_path,
            det_model_dir=model_path,
            # This forces download but doesn't keep the instance
            **{'enable_mkldnn': True}
        )
        print("✅ Models downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        return False

if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)