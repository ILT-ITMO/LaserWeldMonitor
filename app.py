# Root-level app.py - this is what Spaces looks for
import sys
import os

# Add your subfolder to Python path
sys.path.insert(0, 'cv_postprocessing/WeldGUI')

# Import your actual Flask app from the subfolder
from welding.api import app

# This file just redirects to your actual app