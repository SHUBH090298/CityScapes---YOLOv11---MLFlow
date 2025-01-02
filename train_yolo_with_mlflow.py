import os
import yaml
import mlflow
from ultralytics import YOLO
from mlflow_utils import setup_mlflow, log_params, log_metrics, log_artifacts

def load_config(config_path="config.yaml"):
    """
    Loads configuration parameters from a YAML file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def prepare_data_paths(config):
    """
    Prepare the paths for train, val, and test data based on the config.yaml file.
    """
    data_config = config["data"]

    # Directly use the paths provided in config
    train_images = os.path.join(data_config["train"], "images")
    val_images = os.path.join(data_config["val"], "images")
    test_images = os.path.join(data_config["test"], "images")

    # If you need labels, update accordingly
    train_labels = os.path.join(data_config["train"], "labels")
    val_labels = os.path.join(data_config["val"], "labels")

    return {
        "train_images": train_images,
        "val_images": val_images,
        "test_images": test_images,
        "train_labels": train_labels,
        "val_labels": val_labels
    }

def train_yolo_with_mlflow(config):
    """
    Train YOLOv8 with experiment tracking via MLflow.
    """
    setup_mlflow(config["experiment_name"])

    # Extract training parameters
    model_path = config["model"]["name"]
    data_path = config["data"]["yaml_path"]
    training_params = config["training"]
    logging_params = config["logging"]

    # Prepare data paths
    data_paths = prepare_data_paths(config)

    # Log experiment parameters
    log_params({
        "mode": "train",
        "model": model_path,
        "data": data_path,
        **training_params
    })

    # Ensure any previous run is ended
    if mlflow.active_run() is not None:
        mlflow.end_run()

    # Start MLflow tracking
    with mlflow.start_run():
        # Initialize YOLO model with pretrained weights
        model = YOLO(model_path)

        # Train the model with paths for train and val data
        results = model.train(
            data=data_path,
            epochs=training_params["epochs"],
            batch=training_params["batch_size"],
            imgsz=training_params["img_size"],

        )

        # Log metrics
        log_metrics({
            "final_loss": results["metrics"]["box_loss"],
            "map50": results["metrics"]["mAP_50"],
            "map95": results["metrics"]["mAP_50:95"]
        })

        # Log artifacts (optional)
        if logging_params["save_artifacts"]:
            mlflow.log_artifacts("runs/train")

        # Log the final trained model
        if logging_params["log_model"]:
            mlflow.pytorch.log_model(model, artifact_path="model")

def test_yolo_with_mlflow(config):
    """
    Test YOLOv8 with experiment tracking via MLflow.
    """
    setup_mlflow(config["experiment_name"])

    # Extract testing parameters
    model_path = config["model"]["name"]
    data_path = config["data"]["yaml_path"]
    testing_params = config["testing"]
    logging_params = config["logging"]

    # Prepare data paths
    data_paths = prepare_data_paths(config)

    # Log experiment parameters
    log_params({
        "mode": "test",
        "model": model_path,
        "data": data_path,
        **testing_params
    })

    # Ensure any previous run is ended
    if mlflow.active_run() is not None:
        mlflow.end_run()

    # Start MLflow tracking
    with mlflow.start_run():
        # Load the trained YOLO model
        model = YOLO(model_path)

        # Perform validation (no labels needed for test)
        results = model.val(
            data=data_path,
            conf=testing_params["conf_threshold"],
            iou=testing_params["iou_threshold"],
            test_path=data_paths["test_images"]
        )

        # Log metrics
        log_metrics({
            "map50": results["metrics"]["mAP_50"],
            "map95": results["metrics"]["mAP_50:95"]
        })

        # Log artifacts (optional)
        if logging_params["save_artifacts"]:
            mlflow.log_artifacts("runs/val")

def main():
    # Load configuration
    config = load_config("config.yaml")

    # Determine mode and execute the corresponding function
    mode = config.get("mode", "train").lower()
    if mode == "train":
        train_yolo_with_mlflow(config)
    elif mode == "test":
        test_yolo_with_mlflow(config)
    else:
        raise ValueError(f"Invalid mode '{mode}' specified in config. Use 'train' or 'test'.")

if __name__ == "__main__":
    main()
