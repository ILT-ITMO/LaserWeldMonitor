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
import time
import zipfile
import shutil
from predict import main
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'welding/examples/images'


os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OCR once when the app starts
ocr = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)

@app.route('/api/get_scale_params', methods=['POST'])
def get_scale_params():
    # for filename in os.listdir(UPLOAD_FOLDER):
    #     file_path = os.path.join(UPLOAD_FOLDER, filename)
    #     try:
    #         if os.path.isfile(file_path):
    #             os.unlink(file_path)
    #     except Exception as e:
    #         print(f"Error deleting file {file_path}: {e}")
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
        # Handle multiple output files
        processed_files = list(output_dir.glob("*.jpg"))  # Adjust extension if needed
        
        if not processed_files:
            logger.error("No output images found after bulk processing")
            return jsonify({
                "error": "Processing completed but no output images found"
            }), 500
        
        # Create zip file for multiple images
        zip_filename = f"bulk_results_{int(time.time())}.zip"
        zip_path = Path(UPLOAD_FOLDER) / zip_filename
        
        try:
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in processed_files:
                    zipf.write(file, file.name)
        except Exception as e:
            logger.error(f"Failed to create zip file: {str(e)}")
            return jsonify({
                "error": "Failed to package results",
                "details": str(e)
            }), 500
        zip_path = "examples/images/"+zip_filename
        logger.info(f"Successfully processed {len(processed_files)} images")
        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
        
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
    app.run(port=5000)