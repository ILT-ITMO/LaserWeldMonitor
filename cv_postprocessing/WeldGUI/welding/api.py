from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import os
import uuid

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

UPLOAD_FOLDER = 'welding/examples/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    unique_filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)

    result = subprocess.run(
        ["python", "welding/predict.py", file_path],
        capture_output=True,
        text=True
    )

    os.remove(file_path)

    return jsonify({"output": result.stdout})

if __name__ == '__main__':
    app.run(port=5000)
