import mlflow
import os
import yaml

def setup_mlflow(experiment_name, tracking_uri="file:///C:/projects/Cityscapes_Yolov11_MLFlow/runs/mlflow"):
    """
    Sets up MLflow experiment.
    """
    mlflow.set_tracking_uri(tracking_uri)  # Set the URI for MLflow logs
    mlflow.set_experiment(experiment_name)

def log_params(params):
    """
    Logs parameters to MLflow.
    """
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_metrics(metrics):
    """
    Logs metrics to MLflow.
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def log_artifacts(directory):
    """
    Logs artifacts (files, directories) to MLflow.
    """
    if os.path.exists(directory):
        mlflow.log_artifacts(directory)
    else:
        print(f"Warning: The directory {directory} does not exist.")

def load_config(config_path):
    """
    Load configuration from YAML file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def log_model(model, artifact_path):
    """
    Log model to MLflow.
    """
    mlflow.pytorch.log_model(model, artifact_path)

def log_experiment_details(config):
    """
    Logs the experiment details such as model configuration and data details.
    """
    # Log basic experiment details
    mlflow.log_param('experiment_name', config.get('experiment_name'))
    mlflow.log_param('mode', config.get('mode'))

    # Model and training configurations
    model_config = config.get('model', {})
    mlflow.log_param('model_name', model_config.get('name'))
    mlflow.log_param('optimizer', model_config.get('optimizer'))
    mlflow.log_param('learning_rate', model_config.get('learning_rate'))
    mlflow.log_param('weight_decay', model_config.get('weight_decay'))

    scheduler = model_config.get('scheduler', {})
    mlflow.log_param('scheduler_type', scheduler.get('type'))
    mlflow.log_param('warmup_epochs', scheduler.get('warmup_epochs'))

    # Data configuration
    data_config = config.get('data', {})
    mlflow.log_param('data_yaml_path', data_config.get('yaml_path'))
    mlflow.log_param('num_classes', data_config.get('num_classes'))
    mlflow.log_param('val_split', data_config.get('val_split'))

    # Training configuration
    training_config = config.get('training', {})
    mlflow.log_param('epochs', training_config.get('epochs'))
    mlflow.log_param('batch_size', training_config.get('batch_size'))
    mlflow.log_param('img_size', training_config.get('img_size'))
    mlflow.log_param('gradient_accumulation', training_config.get('gradient_accumulation'))
    mlflow.log_param('patience', training_config.get('patience'))
    mlflow.log_param('seed', training_config.get('seed'))

    # Logging configuration
    logging_config = config.get('logging', {})
    mlflow.log_param('save_artifacts', logging_config.get('save_artifacts'))
    mlflow.log_param('log_model', logging_config.get('log_model'))

    # Testing configuration
    testing_config = config.get('testing', {})
    mlflow.log_param('conf_threshold', testing_config.get('conf_threshold'))
    mlflow.log_param('iou_threshold', testing_config.get('iou_threshold'))
    mlflow.log_param('batch_size_test', testing_config.get('batch_size'))

    # Hardware configuration
    hardware_config = config.get('hardware', {})
    mlflow.log_param('use_gpu', hardware_config.get('use_gpu'))
    mlflow.log_param('num_gpus', hardware_config.get('num_gpus'))
    mlflow.log_param('mixed_precision', hardware_config.get('mixed_precision'))

    # Metadata
    metadata = config.get('metadata', {})
    mlflow.log_param('author', metadata.get('author'))
    mlflow.log_param('project', metadata.get('project'))
    mlflow.log_param('description', metadata.get('description'))
    mlflow.log_param('date', metadata.get('date'))

def log_experiment_run(config_path='config.yaml'):
    """
    Main function to load config and log the experiment.
    """
    # Load configuration
    config = load_config(config_path)

    # Set up MLflow experiment
    setup_mlflow(config['experiment_name'])

    # Log experiment details
    log_experiment_details(config)

    # Log parameters, metrics, and artifacts (example usage)
    example_params = {
        'learning_rate': config['model']['learning_rate'],
        'batch_size': config['training']['batch_size']
    }
    log_params(example_params)

    example_metrics = {
        'accuracy': 0.95,
        'loss': 0.05
    }
    log_metrics(example_metrics)

    # Log artifacts directory (assuming you have a directory for saved models or outputs)
    log_artifacts('project/runs')
