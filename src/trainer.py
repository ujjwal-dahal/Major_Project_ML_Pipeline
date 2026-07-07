"""
====================================================
Model Trainer
====================================================

Yo module le ML model train garcha.
Current model = XGBoost

Later SVM use garna yo file matra
change gare pugcha.
"""

from xgboost import XGBClassifier

from config import RANDOM_STATE



def train_xgboost(X_train, y_train):
    """
    XGBoost model train garne function
    """

    print("\nCreating XGBoost Model...")

    # XGBoost model create gareko
    model = XGBClassifier(

        # Total decision trees
        n_estimators=300,

        # Maximum tree depth
        max_depth=6,

        # Learning speed
        learning_rate=0.05,

        # Random sample use garne
        subsample=0.8,

        # Random feature selection
        colsample_bytree=0.8,

        # Same result repeat hos
        random_state=RANDOM_STATE,

        # CPU ko sabai core use garne
        n_jobs=-1,

        # Binary classification evaluation
        eval_metric="logloss"
    )

    print("Training Started...")

    # Model train gareko
    model.fit(X_train, y_train)

    print("Training Completed.")

    return model