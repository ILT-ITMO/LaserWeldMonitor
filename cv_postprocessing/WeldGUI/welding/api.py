# from flask import Flask, jsonify, request, send_file
# from flask_cors import CORS
import gradio as gr
import subprocess
import os
import uuid
import json
import cv2
from paddleocr import PaddleOCR
import numpy as np
from welding.src.ocr import get_pixel_real_size
import traceback
import logging
from pathlib import Path
import sys
import time
import zipfile
import shutil
from welding.predict import main
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# app = Flask(__name__)
# CORS(app)

UPLOAD_FOLDER = 'welding/examples/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# REMOVE THIS: ocr = PaddleOCR(lang="en", use_angle_cls=False)

# Add global variable but don't initialize yet
ocr_instance = None
ocr_lock = threading.Lock()
def get_ocr():
    global ocr_instance
    with ocr_lock:
        if ocr_instance is None:
            print("Initializing PaddleOCR from pre-downloaded models...")
            
            # CORRECTED: Use the actual path from debug output
            model_base = os.path.join(os.getcwd(), 'modelss', 'paddleocr')
            print(f"Looking for models at: {model_base}")
            
            try:
                # Check if models exist before initializing
                required_dirs = [
                    os.path.join(model_base, 'en_PP-OCRv4_det_infer'),
                    os.path.join(model_base, 'en_PP-OCRv4_rec_infer'), 
                    os.path.join(model_base, 'ch_ppocr_mobile_v2.0_cls_infer')
                ]
                
                for model_dir in required_dirs:
                    if not os.path.exists(model_dir):
                        raise FileNotFoundError(f"Model directory not found: {model_dir}")
                    else:
                        print(f"✓ Found model: {model_dir}")
                        print(f"  Contents: {os.listdir(model_dir)}")
                
                # Initialize with explicit paths to pre-downloaded models
                ocr_instance = PaddleOCR(
                    lang="en", 
                    use_angle_cls=False,
                    use_gpu=False,
                    show_log=False,
                    rec_model_dir=os.path.join(model_base, 'en_PP-OCRv4_rec_infer'),
                    det_model_dir=os.path.join(model_base, 'en_PP-OCRv4_det_infer'),
                    cls_model_dir=os.path.join(model_base, 'ch_ppocr_mobile_v2.0_cls_infer'),
                    enable_mkldnn=True
                )
                print("PaddleOCR initialized from pre-downloaded models!")
                
            except Exception as e:
                print(f"Error initializing PaddleOCR with pre-downloaded models: {e}")
                print("Falling back to automatic download...")
                # Use minimal settings to reduce memory footprint
                ocr_instance = PaddleOCR(
                    lang="en", 
                    use_angle_cls=False,
                    use_gpu=False,
                    show_log=False,
                    enable_mkldnn=True
                )
                print("PaddleOCR initialized with automatic download!")
                
        return ocr_instance

def debug_models():
    model_base = '/opt/render/.paddleocr'
    
    if not os.path.exists(model_base):
        return {'error': 'Model directory does not exist'}
    
    model_structure = {}
    for root, dirs, files in os.walk(model_base):
        relative_path = os.path.relpath(root, model_base)
        model_structure[relative_path] = files
    
    return model_structure

def get_scale_params(file):
    try:
        original_filename = Path(file.name).name
        file_path = os.path.join(UPLOAD_FOLDER, original_filename)
        
        with open(file_path, "wb") as f:
            f.write(file.read())

        img = cv2.imread(file_path)
        print("got ocr")
        max_dim = 1024
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        print(f"DEBUG: resized image shape: {img.shape}")

        ocr = get_ocr()
        le, u, line = get_pixel_real_size(ocr, img)
        print(le, "le")

        if line is not None:
            line = [tuple(map(float, point)) for point in line]
        
        return {
            "filename": original_filename,
            "scale_params": {
                "le": float(le),
                "unit": u,
                "reference_line": line
            }
        }

    except Exception as e:
        logger.error(f"Error in get_scale_params: {str(e)}\n{traceback.format_exc()}")
        return {"error": "Internal server error", "details": str(e)}



def health_check():
    return {"status": "healthy", "message": "Flask app is running"}

def process_image(data: dict):
    try:
        # Validate request data
        if not data or 'filename' not in data or 'scale_params' not in data:
            logger.error("Invalid request data")
            return {"error": "Invalid request data"}
        
        print("sggg", data['filename'])
        file_path = os.path.normpath(os.path.join(UPLOAD_FOLDER, data['filename']))
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": "File not found"}
        
        # Prepare config
        config = {
            "image_path": file_path,
            "output_path": "welding/output",
            "middle_part_path": "welding/weights/main.pt",
            "plate_model_path": "welding/weights/plate.pt",
            "render": True,
            "scale_params": data['scale_params']
        }
        
        config_path = os.path.join(UPLOAD_FOLDER, f"config_{data['filename']}.json")
        logger.info(f"Writing config to: {config_path}")
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            logger.error(f"Failed to write config file: {str(e)}")
            return {"error": f"Failed to write config file: {str(e)}"}
   
        results = main(config_dict=config)

        # Return results
        output_base = Path(data['filename']).stem
        rendered_path = Path("welding/output/rendered") / f"{output_base}.jpg"
        print(f"Absolute path being checked: {rendered_path.absolute()}")
        if not rendered_path.exists():
            logger.error(f"Rendered image not found at: {rendered_path}")
            return {
                "error": "Processing completed but output image not found",
                "details": f"Expected path: {rendered_path}"
            }
        
        logger.info("Successfully processed image")
        
        response_data = {
            "success": True,
            "analysis_results": results,
            "image_reference": data['filename'],
            "scale_params": data['scale_params']
        }
        return response_data
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return {
            "error": "Internal server error",
            "details": str(e),
            "traceback": traceback.format_exc()
        }
    finally:
        try:
            if 'config_path' in locals() and os.path.exists(config_path):
                os.remove(config_path)
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")


def process_bulk_images(images, scale_params):
    try:
        output_dir = Path("welding/output/rendered")
        config_path = os.path.join(UPLOAD_FOLDER, f"bulk_config_{int(time.time())}.json")
        config = {
            "image_path": UPLOAD_FOLDER,  # Changed from image_path
            "output_path": "welding/output",
            "middle_part_path": "welding/weights/main.pt",
            "plate_model_path": "welding/weights/plate.pt",
            "render": True,
            "scale_params": scale_params,
            "bulk_process": True  # New flag for bulk processing
        }

        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        if not images:
            return {"error": "No images uploaded"}

        saved_files = []
        for i, img in enumerate(images):
            if img is None:
                continue
            filename = f"bulk_{i}_{uuid.uuid4().hex}.jpg"
            file_path = Path(UPLOAD_FOLDER) / filename
            with open(file_path, "wb") as f:
                f.write(img.read())
            saved_files.append(file_path)

        if not saved_files:
            return {"error": "No valid images provided"}

        try:
            with open(config_path, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            logger.error(f"Failed to write bulk config file: {str(e)}")
            return {"error": f"Failed to write config file: {str(e)}"}
        
        # Run processing
        logger.info("Starting bulk subprocess...")
        results = main(config_dict=config)

        response_data = {
            "success": True,
            "analysis_results": results,
            "scale_params": scale_params
        }

        print(response_data)
        return response_data

    except Exception as e:
        logger.error(f"Unexpected error in bulk processing: {str(e)}\n{traceback.format_exc()}")
        return {
            "error": "Internal server error",
            "details": str(e),
            "traceback": traceback.format_exc()
        }
    finally:
        try:
            if 'config_path' in locals() and os.path.exists(config_path):
                os.remove(config_path)
        except Exception as e:
            logger.error(f"Bulk cleanup failed: {str(e)}")



# Dictionary of your functions
ENDPOINTS = {
    "debug_models": debug_models,
    "get_scale_params": get_scale_params,
    "health_check": health_check,
    "process_image": process_image,
    "process_bulk_images": process_bulk_images,
}


# Create API endpoints using Blocks
with gr.Blocks() as demo:
    gr.Markdown("### 🔧 Backend API Endpoints")
    gr.Interface(fn=debug_models, inputs=[], outputs="json", title="debug_models")
    gr.Interface(fn=get_scale_params, inputs=gr.File(), outputs="json", title="get_scale_params")
    gr.Interface(fn=health_check, inputs=[], outputs="json", title="health_check")
    gr.Interface(fn=process_image, inputs=gr.JSON(), outputs="json", title="process_image")
    gr.Interface(fn=process_bulk_images, inputs=[gr.File(file_count="multiple"), gr.JSON()], outputs="json", title="process_bulk_images")

# Launch on a safe port
try:
    demo.launch(server_name="0.0.0.0", server_port=7860,  share=True)
except Exception as e:
    print(f"Launch failed: {e}")