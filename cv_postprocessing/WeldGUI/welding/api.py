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
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'welding/examples/images'
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
    data = request.json
    if not data or 'filename' not in data or 'scale_params' not in data:
        return jsonify({"error": "Invalid request data"}), 400
    
    file_path = os.path.join('welding', data['filename'])
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    # Save scale params to a temporary config file
    config = {
        "image_path": file_path,
        "output_path": "welding/output",
        "middle_part_path": "welding/models/middle_part.pt",
        "plate_model_path": "welding/models/plate.pt",
        "render": True,
        "scale_params": data['scale_params']
    }
    
    config_path = os.path.join(UPLOAD_FOLDER, f"config_{data['filename']}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # Run processing with the confirmed scale parameters
    result = subprocess.run(
        ["python", "welding/predict.py", "--config", config_path],
        capture_output=True,
        text=True
    )
    
    # Clean up
    os.remove(file_path)
    os.remove(config_path)
    
    if result.returncode != 0:
        return jsonify({"error": result.stderr}), 500
    
    # Return both the measurements and the rendered image
    output_base = os.path.splitext(data['filename'])[0]
    rendered_path = os.path.join("welding/output/rendered", f"{output_base}.jpg")
    
    if not os.path.exists(rendered_path):
        return jsonify({"error": "Processing failed"}), 500
    
    return send_file(rendered_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(port=5000)