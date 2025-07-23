from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import subprocess
import os
import uuid
import json
import cv2
from paddleocr import PaddleOCR
import numpy as np
from src.ocr import get_pixel_real_size
import traceback
import logging
from pathlib import Path
import sys
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'welding/examples/images'
OUTPUT_FOLDER = 'welding/examples/output/rendered/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OCR once when the app starts
ocr = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)

@app.route('/api/get_scale_params', methods=['POST'])
def get_scale_params():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    unique_filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)
    
    # Read the image and get scale parameters
    img = cv2.imread(file_path)
    le, u, line = get_pixel_real_size(ocr, img)
    
    # Convert line points to serializable format
    if line is not None:
        line = [tuple(map(float, point)) for point in line]
    
    # Don't delete the file yet - we'll use it in the next request
    return jsonify({
        "filename": unique_filename,
        "scale_params": {
            "le": float(le),
            "unit": u,
            "reference_line": line
        }
    })

@app.route('/api/process_image', methods=['POST'])
def process_image():
    try:
        data = request.json
        logger.info(f"Received request with data: {data}")
        
        # Validate request data
        if not data or 'filename' not in data or 'scale_params' not in data:
            logger.error("Invalid request data")
            return jsonify({"error": "Invalid request data"}), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, data['filename'])
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
        
        # Run processing
        logger.info("Starting subprocess...")
        try:
            result = subprocess.run(
                ["python", "welding/predict.py", "--config", config_path],
              stdout=sys.stdout, 
    stderr=sys.stderr
            )
            logger.info(f"Subprocess completed with return code: {result.returncode}")
            logger.info(f"Subprocess stdout: {result.stdout}")
            logger.info(f"Subprocess stderr: {result.stderr}")
            
            if result.returncode != 0:
                logger.error(f"Subprocess failed: {result.stderr}")
                return jsonify({
                    "error": "Processing failed",
                    "details": result.stderr
                }), 500
        except Exception as e:
            logger.error(f"Subprocess execution failed: {str(e)}")
            return jsonify({
                "error": "Failed to execute processing",
                "details": str(e)
            }), 500
        
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
        return send_file(str(Path("output/rendered") / f"{output_base}.jpg"), mimetype='image/jpeg')
        
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
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            if 'config_path' in locals() and os.path.exists(config_path):
                os.remove(config_path)
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
if __name__ == '__main__':
    app.run(port=5000)