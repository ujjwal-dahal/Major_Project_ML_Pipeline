"""
====================================================
Project Configuration File
====================================================

यो file मा project मा use hune
common configuration haru rakheko cha.

Later model change, path change,
output folder change garda
yehi file matra modify gare pugcha.
"""

import os

# =====================================
# Dataset Paths
# =====================================

# Training feature CSV
TRAIN_CSV = "data/features_train.csv"

# Development feature CSV
DEV_CSV = "data/features_dev.csv"

# =====================================
# Model Configuration
# =====================================

# Current model name
# XGBoost / SVM / RandomForest / LightGBM
MODEL_NAME = "SVM"

# Model save file name
MODEL_FILE_NAME = f"{MODEL_NAME.lower()}_model.joblib"

# Label Encoder file name
LABEL_ENCODER_FILE = "label_encoder.joblib"

# =====================================
# Output Directories
# =====================================

# Main result folder
RESULT_DIR = os.path.join("results", MODEL_NAME)

# Model save folder
MODEL_DIR = os.path.join(
    RESULT_DIR,
    "models"
)

# CSV save folder
CSV_DIR = os.path.join(
    RESULT_DIR,
    "csv"
)

# Plot save folder
PLOT_DIR = os.path.join(
    RESULT_DIR,
    "plots"
)

# Log save folder
LOG_DIR = os.path.join(
    RESULT_DIR,
    "logs"
)

# =====================================
# Random Seed
# =====================================

# Same result repeat hos bhanera
RANDOM_STATE = 42

# =====================================
# Figure Configuration
# =====================================

# Figure size
FIG_SIZE = (10, 6)

# High resolution image
DPI = 300

# =====================================
# Feature Scaling
# =====================================

# XGBoost lai scaling chaina
# SVM ko lagi True garne
# XGBoost ko lagi scaling OFF cha
USE_SCALER = True

# =====================================
# Train Validation Split
# =====================================

# Official ASVspoof split use bhairaheko
# Future experiments ko lagi matra
TEST_SIZE = 0.2

# =====================================
# Create Folders Automatically
# =====================================

folders = [

    RESULT_DIR,

    MODEL_DIR,

    CSV_DIR,

    PLOT_DIR,

    LOG_DIR

]

for folder in folders:

    os.makedirs(folder, exist_ok=True)