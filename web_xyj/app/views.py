"""Defines a number of routes/views for the flask app."""

from chemai_v1.utils import create_logger, load_task_names, load_args
from chemai_v1.train.run_training import run_training
from chemai_v1.train.make_predictions import make_predictions
from chemai_v1.parsing import add_predict_args, add_train_args, modify_predict_args, modify_train_args
from chemai_v1.data.utils import get_data, get_header, get_smiles, validate_data
from argparse import ArgumentParser, Namespace
from functools import wraps
import io
import os
import sys
import shutil
from tempfile import TemporaryDirectory, NamedTemporaryFile
import time
from typing import Callable, List, Tuple
import multiprocessing as mp
import zipfile

from flask import json, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for

import numpy as np
from rdkit import Chem
from werkzeug.utils import secure_filename

from app import app, db
from flask import Blueprint
from flask_docs import ApiDoc
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))


TRAINING = 0
PROGRESS = mp.Value('d', 0.0)

# Api Document needs to be displayed
app.config['API_DOC_MEMBER'] = ['api']

ApiDoc(app)
api = Blueprint('api', __name__)
app.register_blueprint(api, url_prefix='/')


def check_not_demo(func: Callable) -> Callable:
    """
    View wrapper, which will redirect request to site
    homepage if app is run in DEMO mode.
    :param func: A view which performs sensitive behavior.
    :return: A view with behavior adjusted based on DEMO flag.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if app.config['DEMO']:
            return redirect(url_for('home'))
        return func(*args, **kwargs)

    return decorated_function


def progress_bar(args: Namespace, progress: mp.Value):
    """
    Updates a progress bar displayed during training.

    :param args: Arguments.
    :param progress: The current progress.
    """
    # no code to handle crashes in model training yet, though
    current_epoch = -1
    while current_epoch < args.epochs - 1:
        if os.path.exists(os.path.join(args.save_dir, 'verbose.log')):
            with open(os.path.join(args.save_dir, 'verbose.log'), 'r') as f:
                content = f.read()
                if 'Epoch ' + str(current_epoch + 1) in content:
                    current_epoch += 1
                    progress.value = (current_epoch + 1) * 100 / args.epochs
        else:
            pass
        time.sleep(0)


def find_unused_path(path: str) -> str:
    """
    Given an initial path, finds an unused path by appending different numbers to the filename.

    :param path: An initial path.
    :return: An unused path.
    """
    if not os.path.exists(path):
        return path

    base_name, ext = os.path.splitext(path)

    i = 2
    while os.path.exists(path):
        path = base_name + str(i) + ext
        i += 1

    return path


def name_already_exists_message(thing_being_named: str, original_name: str, new_name: str) -> str:
    """
    Creates a message about a path already existing and therefore being renamed.

    :param thing_being_named: The thing being renamed (ex. Data, Checkpoint).
    :param original_name: The original name of the object.
    :param new_name: The new name of the object.
    :return: A string with a message about the changed name.
    """
    return f'{thing_being_named} "{original_name} already exists. ' \
           f'Saving to "{new_name}".'


def get_upload_warnings_errors(upload_item: str) -> Tuple[List[str], List[str]]:
    """
    Gets any upload warnings passed along in the request.

    :param upload_item: The thing being uploaded (ex. Data, Checkpoint).
    :return: A tuple with a list of warning messages and a list of error messages.
    """
    warnings_raw = request.args.get(f'{upload_item}_upload_warnings')
    errors_raw = request.args.get(f'{upload_item}_upload_errors')
    warnings = json.loads(warnings_raw) if warnings_raw is not None else None
    errors = json.loads(errors_raw) if errors_raw is not None else None

    return warnings, errors


def format_float(value: float, precision: int = 4) -> str:
    """
    Formats a float value to a specific precision.

    :param value: The float value to format.
    :param precision: The number of decimal places to use.
    :return: A string containing the formatted float.
    """
    return f'{value:.{precision}f}'


def format_float_list(array: List[float], precision: int = 4) -> List[str]:
    """
    Formats a list of float values to a specific precision.

    :param array: A list of float values to format.
    :param precision: The number of decimal places to use.
    :return: A list of strings containing the formatted floats.
    """
    return [format_float(f, precision) for f in array]


@app.route('/receiver', methods=['POST'])
@check_not_demo
def receiver():
    """Receiver monitoring the progress of training."""
    return jsonify(progress=PROGRESS.value, training=TRAINING)


@app.route('/')
def home():
    """Renders the home page."""
    return render_template('home.html', users=db.get_all_users())


@app.route('/create_user', methods=['GET', 'POST'])
@check_not_demo
def create_user():
    """
    If a POST request is made, creates a new user.
    Renders the create_user page.
    """
    if request.method == 'GET':
        return render_template('create_user.html', users=db.get_all_users())

    new_name = request.form['newUserName']

    if new_name != None:
        db.insert_user(new_name)

    return redirect(url_for('create_user'))


def render_train(**kwargs):
    """Renders the train page with specified kwargs."""
    data_upload_warnings, data_upload_errors = get_upload_warnings_errors(
        'data')

    return render_template('train.html',
                           datasets=db.get_datasets(
                               request.cookies.get('currentUser')),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           data_upload_warnings=data_upload_warnings,
                           data_upload_errors=data_upload_errors,
                           users=db.get_all_users(),
                           **kwargs)


@app.route('/train', methods=['GET', 'POST'])
@api.route('/train', methods=['GET', 'POST'])
@check_not_demo
def train():
    """Renders the train page and performs training if request method is POST.

    """
    global PROGRESS, TRAINING

    warnings, errors = [], []

    if request.method == 'GET':
        return render_train()

    # Get arguments
    data_name, epochs, ensemble_size, checkpoint_name = \
        request.form['dataName'], int(request.form['epochs']), \
        int(request.form['ensembleSize']), request.form['checkpointName']
    gpu = request.form.get('gpu')
    data_path = os.path.join(app.config['DATA_FOLDER'], f'{data_name}.csv')
    dataset_type = request.form.get('datasetType', 'regression')

    # Create and modify args
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args([])

    args.data_path = data_path
    args.dataset_type = dataset_type
    args.epochs = epochs
    args.ensemble_size = ensemble_size

    # Check if regression/classification selection matches data
    data = get_data(path=data_path)
    targets = data.targets()
    unique_targets = {
        target for row in targets for target in row if target is not None}

    if dataset_type == 'classification' and len(unique_targets - {0, 1}) > 0:
        errors.append(
            'Selected classification dataset but not all labels are 0 or 1. Select regression instead.')

        return render_train(warnings=warnings, errors=errors)

    if dataset_type == 'regression' and unique_targets <= {0, 1}:
        errors.append(
            'Selected regression dataset but all labels are 0 or 1. Select classification instead.')

        return render_train(warnings=warnings, errors=errors)

    if gpu is not None:
        if gpu == 'None':
            args.no_cuda = True
        else:
            args.gpu = int(gpu)

    current_user = request.cookies.get('currentUser')

    if not current_user:
        # Use DEFAULT as current user if the client's cookie is not set.
        current_user = app.config['DEFAULT_USER_ID']

    ckpt_id, ckpt_name = db.insert_ckpt(checkpoint_name,
                                        current_user,
                                        args.dataset_type,
                                        args.epochs,
                                        args.ensemble_size,
                                        len(targets))

    with TemporaryDirectory() as temp_dir:
        args.save_dir = temp_dir
        modify_train_args(args)

        process = mp.Process(target=progress_bar, args=(args, PROGRESS))
        process.start()
        TRAINING = 1

        # Run training
        logger = create_logger(
            name='train', save_dir=args.save_dir, quiet=args.quiet)
        task_scores = run_training(args, logger)
        process.join()

        # Reset globals
        TRAINING = 0
        PROGRESS = mp.Value('d', 0.0)

        # Check if name overlap
        if checkpoint_name != ckpt_name:
            warnings.append(name_already_exists_message(
                'Checkpoint', checkpoint_name, ckpt_name))

        # Move models
        for root, _, files in os.walk(args.save_dir):
            for fname in files:
                if fname.endswith('.pt'):
                    model_id = db.insert_model(ckpt_id)
                    save_path = os.path.join(
                        app.config['CHECKPOINT_FOLDER'], f'{model_id}.pt')
                    shutil.move(os.path.join(
                        args.save_dir, root, fname), save_path)

    return render_train(trained=True,
                        metric=args.metric,
                        num_tasks=len(args.task_names),
                        task_names=args.task_names,
                        task_scores=format_float_list(task_scores),
                        mean_score=format_float(np.mean(task_scores)),
                        warnings=warnings,
                        errors=errors)


def render_predict(**kwargs):
    """Renders the predict page with specified kwargs"""
    checkpoint_upload_warnings, checkpoint_upload_errors = get_upload_warnings_errors(
        'checkpoint')

    return render_template('predict.html',
                           checkpoints=db.get_ckpts(
                               request.cookies.get('currentUser')),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           checkpoint_upload_warnings=checkpoint_upload_warnings,
                           checkpoint_upload_errors=checkpoint_upload_errors,
                           users=db.get_all_users(),
                           **kwargs)


def render_generate(**kwargs):
    """Renders the predict page with specified kwargs"""
    checkpoint_upload_warnings, checkpoint_upload_errors = get_upload_warnings_errors(
        'checkpoint')

    return render_template('generate.html',
                           checkpoints=db.get_ckpts(
                               request.cookies.get('currentUser')),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           checkpoint_upload_warnings=checkpoint_upload_warnings,
                           checkpoint_upload_errors=checkpoint_upload_errors,
                           users=db.get_all_users(),
                           **kwargs)


def render_optimize(**kwargs):
    """Renders the predict page with specified kwargs"""
    checkpoint_upload_warnings, checkpoint_upload_errors = get_upload_warnings_errors(
        'checkpoint')

    return render_template('optimize.html',
                           checkpoints=db.get_ckpts(
                               request.cookies.get('currentUser')),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           checkpoint_upload_warnings=checkpoint_upload_warnings,
                           checkpoint_upload_errors=checkpoint_upload_errors,
                           users=db.get_all_users(),
                           **kwargs)


def render_docking(**kwargs):
    """Renders the predict page with specified kwargs"""
    checkpoint_upload_warnings, checkpoint_upload_errors = get_upload_warnings_errors(
        'checkpoint')

    return render_template('docking.html',
                           checkpoints=db.get_ckpts(
                               request.cookies.get('currentUser')),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           checkpoint_upload_warnings=checkpoint_upload_warnings,
                           checkpoint_upload_errors=checkpoint_upload_errors,
                           users=db.get_all_users(),
                           **kwargs)


def pred_with_one_model(args, ckpt_id, gpu):
    model = db.get_models(ckpt_id)[0]
    model_path = os.path.join(
        app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt')

    task_names = load_task_names(model_path)
    num_tasks = len(task_names)
    args.checkpoint_path = model_path
    if gpu is not None:
        if gpu == 'None':
            args.no_cuda = True
        else:
            args.gpu = int(gpu)
    modify_predict_args(args)
    # Run predictions
    preds = make_predictions(args)
    return preds, num_tasks

@app.route('/predict', methods=['GET', 'POST'])
@api.route('/predict', methods=['GET', 'POST'])
def predict():
    """Renders the predict page and makes predictions if the method is POST.

    @@@
    #### example
    ```
    curl -X POST url  -F 'checkpointName=<checkpointName option>' -F 'textSmiles=<smiles>'
    curl -X POST url -F 'checkpointName=<checkpointName option>' -F 'textSmiles=' -F 'drawSmiles=<drawSmilesInput>'
    curl -X POST url -F 'checkpointName=<checkpointName option>' -F 'textSmiles=' -F 'drawSmiles=' -F 'data=@<fileSmilesInput>'
    ```
    #### return
    - ##### predict.html
    @@@

    """
    if request.method == 'GET':
        return render_predict()

    # Create and modify args
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args([])
    # Get arguments
    ckpt_id = request.form['checkpointName']
    app.config['TEMP_FOLDER'] = './web/web_run_tmp'
    print(ckpt_id)
    args.test_path=app.config['TEMP_FOLDER']


    if request.form['textSmiles'] != '':
        smiles = request.form['textSmiles'].split()
    elif request.form['drawSmiles'] != '':
        smiles = [request.form['drawSmiles']]
    else:
        print(" GOT HERE")
        # Upload data file with SMILES
        data = request.files['data']
        print(data)
        data_name = secure_filename(data.filename)
        data_path = os.path.join(app.config['TEMP_FOLDER'], "raw", data_name)
        data.save(data_path)
        args.preds_path=os.path.join("prediction", data_name)
        args.data_name = data_name.split(".")[0]
        app.config['PREDICTIONS_FILENAME']= args.preds_path

        # Check if header is smiles
        possible_smiles = get_header(data_path)[0]
        smiles = [possible_smiles] if Chem.MolFromSmiles(possible_smiles) is not None else []
        print(data_path)
        # Get remaining smiles
        smiles.extend(get_smiles(data_path))

    print(smiles)
    models = db.get_models(ckpt_id)
    model_paths = [os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt') for model in models]

    task_names = load_task_names(model_paths[0])
    num_tasks = len(task_names)
    gpu = request.form.get('gpu')
    args.checkpoint_paths = model_paths
    if gpu is not None:
        if gpu == 'None':
            args.no_cuda = True
        else:
            args.gpu = int(gpu)
    modify_predict_args(args)
    # Run predictions
    preds = make_predictions(args)

    if all(p is None for p in preds):
        return render_predict(errors=['All SMILES are invalid'])

    # Replace invalid smiles with message
    invalid_smiles_warning = "Invalid SMILES String"
    preds = [pred if pred is not None else [invalid_smiles_warning] * num_tasks for pred in preds]

    return render_predict(predicted=True,
                          smiles=smiles,
                          num_smiles=len(smiles),
                          show_more=0,
                          task_names=task_names,
                          num_tasks=len(task_names),
                          preds=preds,
                          warnings=["List contains invalid SMILES strings"] if None in preds else None,
                          errors=["No SMILES strings given"] if len(preds) == 0 else None)


@app.route('/predict_old', methods=['GET', 'POST'])
@api.route('/predict_old', methods=['GET', 'POST'])
def predict_old():
    """Renders the predict page and makes predictions if the method is POST.

    @@@
    #### example
    ```
    curl -X POST url  -F 'checkpointName=<checkpointName option>' -F 'textSmiles=<smiles>'
    curl -X POST url -F 'checkpointName=<checkpointName option>' -F 'textSmiles=' -F 'drawSmiles=<drawSmilesInput>'
    curl -X POST url -F 'checkpointName=<checkpointName option>' -F 'textSmiles=' -F 'drawSmiles=' -F 'data=@<fileSmilesInput>'
    ```
    #### return
    - ##### predict.html
    @@@

    """
    if request.method == 'GET':
        return render_predict()

    ckpt_id = request.form['checkpointName']
    app.config['TEMP_FOLDER'] = './web/web_log'

    if request.form['textSmiles'] != '':
        smiles = request.form['textSmiles'].split()
    elif request.form['drawSmiles'] != '':
        smiles = [request.form['drawSmiles']]
    else:
        # Upload data file with SMILES
        data = request.files['data']
        data_name = secure_filename(data.filename)
        data_path = os.path.join(app.config['TEMP_FOLDER'], data_name)
        data.save(data_path)

        # Check if header is smiles
        possible_smiles = get_header(data_path)[0]
        smiles = [possible_smiles] if Chem.MolFromSmiles(
            possible_smiles) is not None else []

        # Get remaining smiles
        smiles.extend(get_smiles(data_path))

    # Create and modify args
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args([])

    args.test_path = app.config['TEMP_FOLDER']
    args.data_name = data_name.split(".")[0]
    gpu = request.form.get('gpu')

    model = db.get_models(ckpt_id)[0]
    model_path = os.path.join(
        app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt')

    task_names = load_task_names(model_path)
    num_tasks = len(task_names)

    preds_path = os.path.join(
        app.config['TEMP_FOLDER'], "predict", app.config['PREDICTIONS_FILENAME'])

    args.preds_path = preds_path
    args.checkpoint_paths = model_path
    if gpu is not None:
        if gpu == 'None':
            args.no_cuda = True
        else:
            args.gpu = int(gpu)

    train_args = load_args(model_path)

    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    modify_predict_args(args)

    # Run predictions
    preds = make_predictions(args)

    if all(p is None for p in preds):
        return render_predict(errors=['All SMILES are invalid'])

    # Replace invalid smiles with message
    invalid_smiles_warning = "Invalid SMILES String"
    preds = [pred if pred is not None else [
        invalid_smiles_warning] * num_tasks for pred in preds]

    print(preds)

    return render_predict(predicted=True,
                          smiles=smiles,
                          num_smiles=min(10, len(smiles)),
                          show_more=max(0, len(smiles)-10),
                          task_names=task_names,
                          num_tasks=len(task_names),
                          preds=preds,
                          warnings=[
                              "List contains invalid SMILES strings"] if None in preds else None,
                          errors=["No SMILES strings given"] if len(preds) == 0 else None)


@app.route('/optimize', methods=['GET', 'POST'])
def optimize():
    """Renders the predict page and makes predictions if the method is POST."""
    if request.method == 'GET':
        return render_optimize()

    # Create and modify args
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args([])
    # Get arguments
    ckpt_id = request.form['checkpointName']
    app.config['TEMP_FOLDER'] = '/work01/home/yjxu/documents/A1_IIPharam_project/PKUMDL_ChemAI/test_space'

    args.test_path = app.config['TEMP_FOLDER']

    if request.form['textSmiles'] != '':
        smiles = request.form['textSmiles'].split()
    elif request.form['drawSmiles'] != '':
        smiles = [request.form['drawSmiles']]
    else:
        print(" GOT HERE")
        # Upload data file with SMILES
        data = request.files['data']
        data_name = secure_filename(data.filename)
        data_path = os.path.join(app.config['TEMP_FOLDER'], "raw", data_name)
        data.save(data_path)
        args.preds_path = os.path.join("prediction", data_name)
        args.data_name = data_name.split(".")[0]
        app.config['PREDICTIONS_FILENAME'] = os.path.join(
            "prediction", data_name)

        # Check if header is smiles
        possible_smiles = get_header(data_path)[0]
        smiles = [possible_smiles] if Chem.MolFromSmiles(
            possible_smiles) is not None else []
        print(data_path)
        # Get remaining smiles
        smiles.extend(get_smiles(data_path))

    print(smiles)
    models = db.get_models(ckpt_id)
    model_paths = [os.path.join(
        app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt') for model in models]

    task_names = load_task_names(model_paths[0])
    num_tasks = len(task_names)
    gpu = request.form.get('gpu')
    args.checkpoint_paths = model_paths
    if gpu is not None:
        if gpu == 'None':
            args.no_cuda = True
        else:
            args.gpu = int(gpu)
    modify_predict_args(args)
    # Run predictions
    preds = make_predictions(args)

    if all(p is None for p in preds):
        return render_optimize(errors=['All SMILES are invalid'])

    # Replace invalid smiles with message
    invalid_smiles_warning = "Invalid SMILES String"
    preds = [pred if pred is not None else [
        invalid_smiles_warning] * num_tasks for pred in preds]

    return render_optimize(predicted=True,
                           smiles=smiles,
                           num_smiles=len(smiles),
                           show_more=0,
                           task_names=task_names,
                           num_tasks=len(task_names),
                           preds=preds,
                           warnings=[
                               "List contains invalid SMILES strings"] if None in preds else None,
                           errors=["No SMILES strings given"] if len(preds) == 0 else None)


@app.route('/generate', methods=['GET', 'POST'])
@api.route('/generate', methods=['GET', 'POST'])
def generate():
    """Renders the generate page

    Content-Type: application/json

    args:
        {"checkpointName": "igm_char_rnn, igm_atom_rnn, igm_frag_vae","n_sample_per_model" : "1000", "filter": "None" or "MW_0_1, logP_0_4, SAS_0_5, QED_0_1" }

    return:
        json {"igm_char_rnn": "path_to_charrnn_file", "igm_atom_rnn": "path_to_atomrnn_file","igm_frag_vae": "path_to_fragvae_file", "filtered": "None" or "path_to_filter_file"}
        or
        generate.html

    @@@
    #### example
    ```
    curl -H "Content-Type: application/json" -X POST --data 'args'
    ```

    """
    if request.method == 'GET':
        return render_generate()

    # Get arguments
    print(request.is_json)
    content = request.get_json()
    print(content)

    model_names = content["checkpointName"]
    num_sample = content["n_sample_per_model"]
    isfilter = content["filter"]

    return jsonify({"igm_char_rnn": "path_to_charrnn_file", "igm_atom_rnn": "path_to_atomrnn_file", "igm_frag_vae": "path_to_fragvae_file", "filtered": "None or path_to_filter_file"})

    ckpt_id = request.form['checkpointName']
    app.config['TEMP_FOLDER'] = 'tmp_log_web/generate'
    args.test_path = app.config['TEMP_FOLDER']

    if request.form['textSmiles'] != '':
        smiles = request.form['textSmiles'].split()
    elif request.form['drawSmiles'] != '':
        smiles = [request.form['drawSmiles']]
    else:
        print(" GOT HERE")
        # Upload data file with SMILES
        data = request.files['data']
        data_name = secure_filename(data.filename)
        data_path = os.path.join(app.config['TEMP_FOLDER'], "raw", data_name)
        data.save(data_path)
        args.preds_path = os.path.join("prediction", data_name)
        args.data_name = data_name.split(".")[0]
        app.config['PREDICTIONS_FILENAME'] = os.path.join(
            "prediction", data_name)

        # Check if header is smiles
        possible_smiles = get_header(data_path)[0]
        smiles = [possible_smiles] if Chem.MolFromSmiles(
            possible_smiles) is not None else []
        print(data_path)
        # Get remaining smiles
        smiles.extend(get_smiles(data_path))

    print(smiles)
    models = db.get_models(ckpt_id)
    model_paths = [os.path.join(
        app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt') for model in models]

    task_names = load_task_names(model_paths[0])
    num_tasks = len(task_names)
    gpu = request.form.get('gpu')
    args.checkpoint_paths = model_paths
    if gpu is not None:
        if gpu == 'None':
            args.no_cuda = True
        else:
            args.gpu = int(gpu)
    modify_predict_args(args)
    # Run predictions
    preds = make_predictions(args)

    if all(p is None for p in preds):
        return render_generate(errors=['All SMILES are invalid'])

    # Replace invalid smiles with message
    invalid_smiles_warning = "Invalid SMILES String"
    preds = [pred if pred is not None else [
        invalid_smiles_warning] * num_tasks for pred in preds]

    return render_generate(predicted=True,
                           smiles=smiles,
                           num_smiles=len(smiles),
                           show_more=0,
                           task_names=task_names,
                           num_tasks=len(task_names),
                           preds=preds,
                           warnings=[
                               "List contains invalid SMILES strings"] if None in preds else None,
                           errors=["No SMILES strings given"] if len(preds) == 0 else None)


@app.route('/docking', methods=['GET', 'POST'])
def docking():
    """Renders the predict page and makes predictions if the method is POST."""
    if request.method == 'GET':
        return render_docking()

    # Create and modify args
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args([])
    # Get arguments
    ckpt_id = request.form['checkpointName']
    app.config['TEMP_FOLDER'] = '/work01/home/yjxu/documents/A1_IIPharam_project/PKUMDL_ChemAI/test_space'

    args.test_path = app.config['TEMP_FOLDER']

    if request.form['textSmiles'] != '':
        smiles = request.form['textSmiles'].split()
    elif request.form['drawSmiles'] != '':
        smiles = [request.form['drawSmiles']]
    else:
        print(" GOT HERE")
        # Upload data file with SMILES
        data = request.files['data']
        data_name = secure_filename(data.filename)
        data_path = os.path.join(app.config['TEMP_FOLDER'], "raw", data_name)
        data.save(data_path)
        args.preds_path = os.path.join("prediction", data_name)
        args.data_name = data_name.split(".")[0]
        app.config['PREDICTIONS_FILENAME'] = os.path.join(
            "prediction", data_name)

        # Check if header is smiles
        possible_smiles = get_header(data_path)[0]
        smiles = [possible_smiles] if Chem.MolFromSmiles(
            possible_smiles) is not None else []
        print(data_path)
        # Get remaining smiles
        smiles.extend(get_smiles(data_path))

    print(smiles)
    models = db.get_models(ckpt_id)
    model_paths = [os.path.join(
        app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt') for model in models]

    task_names = load_task_names(model_paths[0])
    num_tasks = len(task_names)
    gpu = request.form.get('gpu')
    args.checkpoint_paths = model_paths
    if gpu is not None:
        if gpu == 'None':
            args.no_cuda = True
        else:
            args.gpu = int(gpu)
    modify_predict_args(args)
    # Run predictions
    preds = make_predictions(args)

    if all(p is None for p in preds):
        return render_docking(errors=['All SMILES are invalid'])

    # Replace invalid smiles with message
    invalid_smiles_warning = "Invalid SMILES String"
    preds = [pred if pred is not None else [
        invalid_smiles_warning] * num_tasks for pred in preds]

    return render_docking(predicted=True,
                          smiles=smiles,
                          num_smiles=len(smiles),
                          show_more=0,
                          task_names=task_names,
                          num_tasks=len(task_names),
                          preds=preds,
                          warnings=[
                              "List contains invalid SMILES strings"] if None in preds else None,
                          errors=["No SMILES strings given"] if len(preds) == 0 else None)


@app.route('/download_predictions')
def download_predictions():
    """Downloads predictions as a .csv file."""
    return send_from_directory(app.config['TEMP_FOLDER'], app.config['PREDICTIONS_FILENAME'], as_attachment=True, cache_timeout=-1)


@app.route('/data')
@check_not_demo
def data():
    """Renders the data page."""
    data_upload_warnings, data_upload_errors = get_upload_warnings_errors(
        'data')

    return render_template('data.html',
                           datasets=db.get_datasets(
                               request.cookies.get('currentUser')),
                           data_upload_warnings=data_upload_warnings,
                           data_upload_errors=data_upload_errors,
                           users=db.get_all_users())


@app.route('/data/upload/<string:return_page>', methods=['POST'])
@check_not_demo
def upload_data(return_page: str):
    """
    Uploads a data .csv file.

    :param return_page: The name of the page to render to after uploading the dataset.
    """
    warnings, errors = [], []

    current_user = request.cookies.get('currentUser')

    if not current_user:
        # Use DEFAULT as current user if the client's cookie is not set.
        current_user = app.config['DEFAULT_USER_ID']

    dataset = request.files['dataset']

    with NamedTemporaryFile() as temp_file:
        dataset.save(temp_file.name)
        dataset_errors = validate_data(temp_file.name)

        if len(dataset_errors) > 0:
            errors.extend(dataset_errors)
        else:
            dataset_name = request.form['datasetName']
            # dataset_class = load_args(ckpt).dataset_type  # TODO: SWITCH TO ACTUALLY FINDING THE CLASS

            dataset_id, new_dataset_name = db.insert_dataset(
                dataset_name, current_user, 'UNKNOWN')

            dataset_path = os.path.join(
                app.config['DATA_FOLDER'], f'{dataset_id}.csv')

            if dataset_name != new_dataset_name:
                warnings.append(name_already_exists_message(
                    'Data', dataset_name, new_dataset_name))

            shutil.copy(temp_file.name, dataset_path)

    warnings, errors = json.dumps(warnings), json.dumps(errors)

    return redirect(url_for(return_page, data_upload_warnings=warnings, data_upload_errors=errors))


@app.route('/data/download/<int:dataset>')
@check_not_demo
def download_data(dataset: int):
    """
    Downloads a dataset as a .csv file.

    :param dataset: The id of the dataset to download.
    """
    return send_from_directory(app.config['DATA_FOLDER'], f'{dataset}.csv', as_attachment=True, cache_timeout=-1)


@app.route('/data/delete/<int:dataset>')
@check_not_demo
def delete_data(dataset: int):
    """
    Deletes a dataset.

    :param dataset: The id of the dataset to delete.
    """
    db.delete_dataset(dataset)
    os.remove(os.path.join(app.config['DATA_FOLDER'], f'{dataset}.csv'))
    return redirect(url_for('data'))


@app.route('/checkpoints')
@check_not_demo
def checkpoints():
    """Renders the checkpoints page."""
    checkpoint_upload_warnings, checkpoint_upload_errors = get_upload_warnings_errors(
        'checkpoint')

    return render_template('checkpoints.html',
                           checkpoints=db.get_ckpts(
                               request.cookies.get('currentUser')),
                           checkpoint_upload_warnings=checkpoint_upload_warnings,
                           checkpoint_upload_errors=checkpoint_upload_errors,
                           users=db.get_all_users())


@app.route('/checkpoints/upload/<string:return_page>', methods=['POST'])
@check_not_demo
def upload_checkpoint(return_page: str):
    """
    Uploads a checkpoint .pt file.

    :param return_page: The name of the page to render after uploading the checkpoint file.
    """
    warnings, errors = [], []

    current_user = request.cookies.get('currentUser')

    if not current_user:
        # Use DEFAULT as current user if the client's cookie is not set.
        current_user = app.config['DEFAULT_USER_ID']

    ckpt = request.files['checkpoint']

    ckpt_name = request.form['checkpointName']

    # Create temporary file to get ckpt_args without losing data.
    with NamedTemporaryFile() as temp_file:
        ckpt.save(temp_file.name)

        ckpt_args = load_args(temp_file)

        ckpt_id, new_ckpt_name = db.insert_ckpt(ckpt_name,
                                                current_user,
                                                ckpt_args.dataset_type,
                                                ckpt_args.epochs,
                                                1,
                                                ckpt_args.train_data_size)

        model_id = db.insert_model(ckpt_id)

        model_path = os.path.join(
            app.config['CHECKPOINT_FOLDER'], f'{model_id}.pt')

        if ckpt_name != new_ckpt_name:
            warnings.append(name_already_exists_message(
                'Checkpoint', ckpt_name, new_ckpt_name))

        shutil.copy(temp_file.name, model_path)

    warnings, errors = json.dumps(warnings), json.dumps(errors)

    return redirect(url_for(return_page, checkpoint_upload_warnings=warnings, checkpoint_upload_errors=errors))


@app.route('/checkpoints/download/<int:checkpoint>')
@check_not_demo
def download_checkpoint(checkpoint: int):
    """
    Downloads a zip of model .pt files.

    :param checkpoint: The name of the checkpoint to download.
    """
    ckpt = db.query_db(f'SELECT * FROM ckpt WHERE id = {checkpoint}', one=True)
    models = db.get_models(checkpoint)

    model_data = io.BytesIO()

    with zipfile.ZipFile(model_data, mode='w') as z:
        for model in models:
            model_path = os.path.join(
                app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt')
            z.write(model_path, os.path.basename(model_path))

    model_data.seek(0)

    return send_file(
        model_data,
        mimetype='application/zip',
        as_attachment=True,
        attachment_filename=f'{ckpt["ckpt_name"]}.zip',
        cache_timeout=-1
    )


@app.route('/checkpoints/delete/<int:checkpoint>')
@check_not_demo
def delete_checkpoint(checkpoint: int):
    """
    Deletes a checkpoint file.

    :param checkpoint: The id of the checkpoint to delete.
    """
    db.delete_ckpt(checkpoint)
    return redirect(url_for('checkpoints'))


app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)