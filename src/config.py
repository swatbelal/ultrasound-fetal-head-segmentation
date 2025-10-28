#!/usr/bin/env python3
from pathlib import Path

# ---------------------------
# Project Root
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------
# Dataset Paths
# ---------------------------
DATA_DIR = PROJECT_ROOT / "data" / "fetal-head-ultrasound"

TRAIN_IMG_DIR = DATA_DIR / "train" / "img"
TRAIN_MASK_DIR = DATA_DIR / "train" / "ann"
TRAIN_MASK_IMG_DIR = DATA_DIR / "train" / "masks_png"

VAL_IMG_DIR = DATA_DIR / "test" / "img"
VAL_MASK_DIR = DATA_DIR / "test" / "ann"
VAL_MASK_IMG_DIR = DATA_DIR / "test" / "masks_png"

# ---------------------------
# Training Parameters
# ---------------------------
IMG_SIZE = (256, 256)         # input size for U-Net
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4

# ---------------------------
# Results / Outputs
# ---------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = RESULTS_DIR / "unet_fetal_head.keras"
HISTORY_PATH = RESULTS_DIR / "history.txt"
EVAL_LOCK_FILE = RESULTS_DIR / "eval.lock"

# ---------------------------
# Logging
# ---------------------------
LOG_FILE_NAME = RESULTS_DIR / 'eval.log'
# Log level can be: debug, info, warning, error, critical
LOG_LEVEL = 'info'
