from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
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
app = Flask(__name__)
CORS(app)

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
            model_base = os.path.join(os.getcwd(), 'models', 'paddleocr')
            print(f"Looking for models at: {model_base}")
            
            try:
                # Check if models exist before initializing
                required_dirs = [
                    os.path.join(model_base, 'en_PP-OCRv3_det_infer'),
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
                    det_model_dir=os.path.join(model_base, 'en_PP-OCRv3_det_infer'),
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

@app.route('/api/get_scale_params', methods=['POST'])
def get_scale_params():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        original_filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(file_path)
        
        # Read the image and get scale parameters
        img = cv2.imread(file_path)
        
        # LAZY INITIALIZATION - only when needed
        ocr = get_ocr()
        le, u, line = get_pixel_real_size(ocr, img)
        
        # Convert line points to serializable format
        if line is not None:
            line = [tuple(map(float, point)) for point in line]
        
        return jsonify({
            "filename": original_filename,
            "scale_params": {
                "le": float(le),
                "unit": u,
                "reference_line": line
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_scale_params: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with filesystem debugging"""
    debug_info = {
        "status": "healthy",
        "message": "Flask app is running",
        "filesystem_debug": {}
    }
    
    try:
        # Current working directory info
        debug_info["filesystem_debug"]["current_directory"] = os.getcwd()
        debug_info["filesystem_debug"]["directory_contents"] = os.listdir('.')
        
        # Check models directory
        models_path = os.path.join(os.getcwd(), 'models')
        debug_info["filesystem_debug"]["models_directory_exists"] = os.path.exists(models_path)
        
        if os.path.exists(models_path):
            debug_info["filesystem_debug"]["models_directory_contents"] = os.listdir(models_path)
            
            # Check paddleocr subdirectory
            paddleocr_path = os.path.join(models_path, 'paddleocr')
            debug_info["filesystem_debug"]["paddleocr_directory_exists"] = os.path.exists(paddleocr_path)
            
            if os.path.exists(paddleocr_path):
                debug_info["filesystem_debug"]["paddleocr_directory_contents"] = os.listdir(paddleocr_path)
                
                # Check each model directory
                model_dirs = ['en_PP-OCRv3_det_infer', 'en_PP-OCRv4_rec_infer', 'ch_ppocr_mobile_v2.0_cls_infer']
                for model_dir in model_dirs:
                    model_path = os.path.join(paddleocr_path, model_dir)
                    debug_info["filesystem_debug"][f"{model_dir}_exists"] = os.path.exists(model_path)
                    if os.path.exists(model_path):
                        debug_info["filesystem_debug"][f"{model_dir}_contents"] = os.listdir(model_path)
        
        # Check absolute paths that might be used on Render
        abs_paths_to_check = [
            '/opt/render/project/src',
            '/app',
            '/var/task'
        ]
        
        debug_info["filesystem_debug"]["absolute_paths"] = {}
        for abs_path in abs_paths_to_check:
            debug_info["filesystem_debug"]["absolute_paths"][abs_path] = {
                "exists": os.path.exists(abs_path),
                "is_directory": os.path.isdir(abs_path) if os.path.exists(abs_path) else False
            }
            if os.path.exists(abs_path) and os.path.isdir(abs_path):
                try:
                    debug_info["filesystem_debug"]["absolute_paths"][abs_path]["contents"] = os.listdir(abs_path)[:10]  # First 10 items
                except PermissionError:
                    debug_info["filesystem_debug"]["absolute_paths"][abs_path]["contents"] = "Permission denied"
        
        # Environment variables (for debugging)
        debug_info["environment"] = {
            "PYTHONPATH": os.environ.get('PYTHONPATH'),
            "PWD": os.environ.get('PWD'),
            "HOME": os.environ.get('HOME')
        }
        
    except Exception as e:
        debug_info["status"] = "error"
        debug_info["error"] = str(e)
    
    return jsonify(debug_info)


@app.route('/api/debug/models', methods=['GET'])
def debug_models():
    """Specific endpoint to debug model paths"""
    debug_info = {}
    
    # Test all possible model path locations
    possible_base_paths = [
        os.getcwd(),
        '/opt/render/project/src',
        '/app',
        '/var/task',
        '/'
    ]
    
    model_dirs = ['en_PP-OCRv3_det_infer', 'en_PP-OCRv4_rec_infer', 'ch_ppocr_mobile_v2.0_cls_infer']
    
    debug_info["possible_paths"] = {}
    
    for base_path in possible_base_paths:
        if os.path.exists(base_path):
            models_path = os.path.join(base_path, 'models', 'paddleocr')
            debug_info["possible_paths"][base_path] = {
                "base_exists": True,
                "models_paddleocr_exists": os.path.exists(models_path),
                "models": {}
            }
            
            if os.path.exists(models_path):
                for model_dir in model_dirs:
                    model_path = os.path.join(models_path, model_dir)
                    debug_info["possible_paths"][base_path]["models"][model_dir] = {
                        "exists": os.path.exists(model_path),
                        "path": model_path
                    }
                    if os.path.exists(model_path):
                        debug_info["possible_paths"][base_path]["models"][model_dir]["contents"] = os.listdir(model_path)
        else:
            debug_info["possible_paths"][base_path] = {"base_exists": False}
    
    return jsonify(debug_info)
@app.route('/api/process_image', methods=['POST'])
def process_image():
    try:
        data = request.json
        logger.info(f"Received request with data: {data}")
        
        # Validate request data
        if not data or 'filename' not in data or 'scale_params' not in data:
            logger.error("Invalid request data")
            return jsonify({"error": "Invalid request data"}), 400
        print("sggg",data['filename'])
        file_path = os.path.normpath(os.path.join(UPLOAD_FOLDER, data['filename']))
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({"error": "File not found"}), 404
        
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
            return jsonify({"error": f"Failed to write config file: {str(e)}"}), 500
   
        results = main(config_dict=config)
        # print(results)
        # Return results
        output_base = Path(data['filename']).stem
        rendered_path = Path("welding/output/rendered") / f"{output_base}.jpg"
        print(f"Absolute path being checked: {rendered_path.absolute()}")
        if not rendered_path.exists():
            logger.error(f"Rendered image not found at: {rendered_path}")
            return jsonify({
                "error": "Processing completed but output image not found",
                "details": f"Expected path: {rendered_path}"
            }), 500
        
        logger.info("Successfully processed image")
        
        response_data = {
            "success": True,
            "analysis_results": results,  # All your measurement data
            "image_reference": data['filename'],  # Same reference you received
            "scale_params": data['scale_params']  # Return the scale params back for verification
        }
        # print(response_data)
        return jsonify(response_data)
        # return send_file(str(Path("output/rendered") / f"{output_base}.jpg"), mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500
    finally:
        # Clean up in finally block to ensure it runs even if there's an error
        try:
            # if 'file_path' in locals() and os.path.exists(file_path):
            #     os.remove(file_path)
            if 'config_path' in locals() and os.path.exists(config_path):
                os.remove(config_path)
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
@app.route('/api/process_bulk_images', methods=['POST'])
def process_bulk_images():
    try:
        images = request.files.getlist('images')
    
    # Get scale params
        scale_params = json.loads(request.form.get('scale_params'))
     
        # Prepare config (different from single image)

        
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
        images = request.files.getlist('images')
        if not images:
            return jsonify({"error": "No images uploaded"}), 400

        # Save all uploaded images to UPLOAD_FOLDER
        saved_files = []
        for img in images:
            if img.filename == '':
                continue
            filename = img.filename
            file_path = Path(UPLOAD_FOLDER) / filename
            img.save(file_path)
            saved_files.append(file_path)

        if not saved_files:
            return jsonify({"error": "No valid images provided"}), 400

        try:
            with open(config_path, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            logger.error(f"Failed to write bulk config file: {str(e)}")
            return jsonify({"error": f"Failed to write config file: {str(e)}"}), 500
        
        # Run processing
        logger.info("Starting bulk subprocess...")
   
        results = main(config_dict=config)
        response_data = {
            "success": True,
            "analysis_results": results,  # All your measurement data
            # "image_reference": data['filename'],  # Same reference you received
            "scale_params": scale_params  # Return the scale params back for verification
        }
        print(response_data)
        # print(response_data)
        return jsonify(response_data)

        
    except Exception as e:
        logger.error(f"Unexpected error in bulk processing: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500
    finally:
        # Cleanup
        try:
            if 'config_path' in locals() and os.path.exists(config_path):
                os.remove(config_path)
            # Don't delete the input folder, just cleanup config
        except Exception as e:
            logger.error(f"Bulk cleanup failed: {str(e)}")
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)