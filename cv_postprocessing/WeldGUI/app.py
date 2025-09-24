# WeldGUI/app.py
from welding.api import app  # Import your Flask app from the welding package

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)