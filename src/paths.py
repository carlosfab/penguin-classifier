# Path: src/paths.py
# --- Imports ---
import os
from pathlib import Path

# --- Constants ---
PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / "data"
CODE_FOLDER = PARENT_DIR / "src"
INFERENCE_CODE_FOLDER = CODE_FOLDER / "inference"
