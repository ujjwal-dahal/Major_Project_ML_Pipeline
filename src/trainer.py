"""
====================================================
Model Trainer
====================================================

Yo module le selected ML model train garcha.

Supported Models

- XGBoost
- SVM
- Random Forest
"""

from config import MODEL_NAME
from config import RANDOM_STATE

# XGBoost
from xgboost import XGBClassifier

# SVM
from sklearn.svm import SVC

# Random Forest
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Config file bata model select garera
    automatically train garne function
    """

    # =====================================
    # XGBoost
    # =====================================

    if MODEL_NAME == "XGBoost":

        print("\nCreating XGBoost Model...")

        model = XGBClassifier(

            n_estimators=300,

            max_depth=6,

            learning_rate=0.05,

            subsample=0.8,

            colsample_bytree=0.8,

            random_state=RANDOM_STATE,

            n_jobs=-1,

            eval_metric="logloss"

        )

    # =====================================
    # SVM
    # =====================================

    elif MODEL_NAME == "SVM":

        print("\nCreating SVM Model...")

        model = SVC(

            kernel="rbf",

            C=1.0,

            gamma="scale",

            probability=True,

            random_state=RANDOM_STATE

        )

    # =====================================
    # Random Forest
    # =====================================

    elif MODEL_NAME == "RandomForest":

        print("\nCreating Random Forest Model...")

        model = RandomForestClassifier(

            n_estimators=300,

            max_depth=None,

            random_state=RANDOM_STATE,

            n_jobs=-1

        )

    # =====================================
    # Invalid Model
    # =====================================

    else:

        raise ValueError(

            f"{MODEL_NAME} is not supported."

        )

    print("Training Started...")

    # Model train gareko
    model.fit(

        X_train,

        y_train

    )

    print("Training Completed.")

    return model