import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

project_name = "doclens"

list_of_files = [
    "README.md",
    "requirements.txt",
    "setup.py",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/search.py",
    f"src/{project_name}/sentiment.py",
    f"src/{project_name}/clustering.py",
    f"src/{project_name}/visualization.py",
    f"src/{project_name}/utils.py",
    "notebooks/exploration.ipynb",
    "data/.gitkeep",
    "results/.gitkeep",
    "tests/__init__.py",
    "tests/test_search.py",
    "tests/test_sentiment.py",
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, filename = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Created directory: {file_dir} for file: {filename}")

    if (not file_path.exists()) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass
        logging.info(f"Created file: {file_path}")
    else:
        logging.info(f"{filename} already exists.")
