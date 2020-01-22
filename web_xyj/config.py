"""
Sets the config parameters for the flask app object.
These are accessible in a dictionary, with each line defining a key.
"""

import os
from tempfile import TemporaryDirectory

import torch

_TEMP_FOLDER_OBJECT = TemporaryDirectory()

DEFAULT_USER_ID = 1
ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'app/web_data')
CHECKPOINT_FOLDER = os.path.join(ROOT_FOLDER, 'app/web_checkpoints')
TEMP_FOLDER = os.path.join(ROOT_FOLDER, 'web_run_tmp')
SMILES_FILENAME = 'smiles.csv'
PREDICTIONS_FILENAME = 'predictions.csv'
GENERATIONS_FILENAME = 'generations.smi'

DB_FILENAME = 'chemprop.sqlite3'
CUDA = torch.cuda.is_available()
GPUS = list(range(torch.cuda.device_count()))

os.makedirs(os.path.join(TEMP_FOLDER, "raw"), exist_ok=True)
os.makedirs(os.path.join(TEMP_FOLDER, "prediction"), exist_ok=True)
os.makedirs(os.path.join(TEMP_FOLDER, "generation"), exist_ok=True)
os.makedirs(os.path.join(TEMP_FOLDER, "optimization"), exist_ok=True)
os.makedirs(os.path.join(TEMP_FOLDER, "docking"), exist_ok=True)