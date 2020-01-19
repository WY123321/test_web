"""Runs the web interface version of chemprop, allowing for training and predicting in a web browser."""
import os
from flask import Flask



app = Flask(__name__)

app.config.from_object('config')

os.makedirs(app.config['CHECKPOINT_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

# Local loading
# app.config['API_DOC_CDN'] = False

# Disable document pages
# app.config['API_DOC_ENABLE'] = False



from app import views
