import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    ".github/workflows/.gitkeep",
    "train_yolo_with_mlflow.py",
    "config.yaml",
    "data.yaml",
    "mlflow_utils.py",
    "runs/",  # Ensures it's a directory
    "train/",  # Ensures it's a directory
    "val/",  # Ensures it's a directory
    "constants/__init__.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)

    if filepath.suffix == "":  # If no file extension, treat as a directory
        if not filepath.exists():
            os.makedirs(filepath, exist_ok=True)
            logging.info(f"Creating directory: {filepath}")
        else:
            logging.info(f"Directory already exists: {filepath}")
    else:
        # If it's a file, ensure parent directory exists
        filedir = filepath.parent
        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Ensuring directory exists: {filedir}")

        # Create the file if it doesn't exist
        if not filepath.exists():
            with open(filepath, "w") as f:
                pass
            logging.info(f"Creating empty file: {filepath}")
        else:
            logging.info(f"File already exists: {filepath}")
