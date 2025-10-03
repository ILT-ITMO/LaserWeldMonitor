import os
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "Hello from Flask on Hugging Face!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # fallback to 7860 if PORT is unset
    app.run(host="0.0.0.0", port=port)
