"""
====================================================
Model Evaluation Module
====================================================

Yo module le

1. Prediction garcha
2. Probability prediction garcha
3. Evaluation metrics calculate garcha
4. CSV save garcha
"""

import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score
)

from config import CSV_DIR


def evaluate_model(model,
                   X_valid,
                   y_valid):
    """
    Trained model evaluate garne function
    """

    print("\nMaking Predictions...")

    # Validation prediction gareko
    y_pred = model.predict(X_valid)

    # Probability prediction gareko
    # Probability Prediction bhaneko chai model le kun class ma belong garcha bhanne probability nikalne ho
    y_prob = model.predict_proba(X_valid)[:, 1]

    print("Prediction Completed.")

    # ==========================================
    # Confusion Matrix
    # ==========================================

    # Confusion Matrix calculate gareko
    cm = confusion_matrix(y_valid, y_pred)

    # Matrix bata values nikaleko
    TN, FP, FN, TP = cm.ravel()

    # ==========================================
    # Basic Metrics
    # ==========================================

    accuracy = accuracy_score(y_valid, y_pred)

    precision = precision_score(
        y_valid,
        y_pred
    )

    recall = recall_score(
        y_valid,
        y_pred
    )

    f1 = f1_score(
        y_valid,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_valid,
        y_prob
    )

    balanced_acc = balanced_accuracy_score(
        y_valid,
        y_pred
    )

    logloss = log_loss(
        y_valid,
        y_prob
    )

    mcc = matthews_corrcoef(
        y_valid,
        y_pred
    )

    kappa = cohen_kappa_score(
        y_valid,
        y_pred
    )

    # ==========================================
    # Manual Metrics
    # ==========================================

    # Sensitivity = Recall
    sensitivity = TP / (TP + FN)

    # Specificity
    specificity = TN / (TN + FP)

    # False Positive Rate
    fpr = FP / (FP + TN)

    # False Negative Rate
    fnr = FN / (FN + TP)

    # True Positive Rate
    tpr = sensitivity

    # True Negative Rate
    tnr = specificity

    # ==========================================
    # Print Metrics
    # ==========================================

    print("\n==============================")
    print("Evaluation Metrics")
    print("==============================")

    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1 Score           : {f1:.4f}")
    print(f"ROC AUC            : {roc_auc:.4f}")
    print(f"Balanced Accuracy  : {balanced_acc:.4f}")
    print(f"Sensitivity        : {sensitivity:.4f}")
    print(f"Specificity        : {specificity:.4f}")
    print(f"FPR                : {fpr:.4f}")
    print(f"FNR                : {fnr:.4f}")
    print(f"TPR                : {tpr:.4f}")
    print(f"TNR                : {tnr:.4f}")
    print(f"Log Loss           : {logloss:.4f}")
    print(f"MCC                : {mcc:.4f}")
    print(f"Cohen Kappa        : {kappa:.4f}")

    # ==========================================
    # Classification Report
    # ==========================================

    print("\nClassification Report\n")

    print(
    classification_report(
        y_valid,
        y_pred,
        target_names=["bonafide", "spoof"]
      )
    )

    # ==========================================
    # Save Evaluation Metrics
    # ==========================================

    metrics = {

        "Metric": [

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score",

            "ROC AUC",

            "Balanced Accuracy",

            "Sensitivity",

            "Specificity",

            "False Positive Rate",

            "False Negative Rate",

            "True Positive Rate",

            "True Negative Rate",

            "Log Loss",

            "Matthews Correlation Coefficient",

            "Cohen Kappa"

        ],

        "Value": [

            accuracy,

            precision,

            recall,

            f1,

            roc_auc,

            balanced_acc,

            sensitivity,

            specificity,

            fpr,

            fnr,

            tpr,

            tnr,

            logloss,

            mcc,

            kappa

        ]

    }

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(

        os.path.join(

            CSV_DIR,

            "evaluation_metrics.csv"

        ),

        index=False

    )

    # ==========================================
    # Save Predictions
    # ==========================================

    prediction_df = pd.DataFrame({

        "Actual": y_valid,

        "Prediction": y_pred,

        "Probability": y_prob

    })

    prediction_df.to_csv(

        os.path.join(

            CSV_DIR,

            "validation_predictions.csv"

        ),

        index=False

    )

    print("\nCSV Files Saved Successfully.")

    return {

        "y_pred": y_pred,

        "y_prob": y_prob,

        "confusion_matrix": cm,

        "metrics": metrics_df

    }