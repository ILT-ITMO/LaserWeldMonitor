FROM python:3.9-slim

# set workdir
WORKDIR /app

# copy dependencies
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy all project files
COPY . .

# Hugging Face sets $PORT dynamically
# Use gunicorn to serve the Flask app
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:7860 cv_postprocessing.WeldGUI.app:app"]
