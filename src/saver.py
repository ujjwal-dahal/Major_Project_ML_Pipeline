"""
====================================================
Model Saving Module
====================================================

Yo module le

1. Model save garcha
2. Label Encoder save garcha
"""

import os
import joblib

from config import MODEL_DIR
from config import MODEL_FILE_NAME
from config import LABEL_ENCODER_FILE


def save_model(model):
    """
    Trained model save garne function
    """

    # Model save path
    model_path = os.path.join(

        MODEL_DIR,

        MODEL_FILE_NAME

    )

    # Model save gareko
    joblib.dump(

        model,

        model_path

    )

    print(f"Model Saved : {model_path}")


def save_label_encoder(label_encoder):
    """
    Label Encoder save garne function
    """

    # Label encoder save path
    encoder_path = os.path.join(

        MODEL_DIR,

        LABEL_ENCODER_FILE

    )

    # Encoder save gareko
    joblib.dump(

        label_encoder,

        encoder_path

    )

    print(f"Label Encoder Saved : {encoder_path}")