"""
====================================================
Visualization Module
====================================================

Yo module le

1. Confusion Matrix Heatmap
2. ROC Curve
3. Precision Recall Curve

generate garcha.
"""

import os

import matplotlib

# GUI backend use nagari image file matra save garne
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc
)

from config import PLOT_DIR
from config import CSV_DIR
from config import DPI
from config import FIG_SIZE

def plot_confusion_matrix(confusion_matrix):
    """
    Confusion Matrix Heatmap
    """

    plt.figure(figsize=FIG_SIZE)

    # Heatmap draw gareko
    sns.heatmap(

        confusion_matrix,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=["Bonafide", "Spoof"],

        yticklabels=["Bonafide", "Spoof"]

    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted Label")

    plt.ylabel("True Label")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOT_DIR,

            "confusion_matrix.png"

        ),

        dpi=DPI

    )

    plt.close()

    print("Confusion Matrix Saved.")
    

# ROC Curve generate garne function    
def plot_roc_curve(
        y_true,
        y_probability):
    """
    ROC Curve generate garne
    """

    # ROC values calculate gareko
    fpr, tpr, thresholds = roc_curve(

        y_true,

        y_probability

    )

    # AUC calculate gareko
    roc_auc = auc(

        fpr,

        tpr

    )

    # =====================================
    # Save CSV
    # =====================================

    roc_df = pd.DataFrame({

        "FPR": fpr,

        "TPR": tpr,

        "Threshold": thresholds

    })

    roc_df.to_csv(

        os.path.join(

            CSV_DIR,

            "roc_curve_values.csv"

        ),

        index=False

    )

    # =====================================
    # Plot
    # =====================================

    plt.figure(figsize=FIG_SIZE)

    plt.plot(

        fpr,

        tpr,

        linewidth=2,

        label=f"AUC = {roc_auc:.4f}"

    )

    # Random classifier line
    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--"

    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOT_DIR,

            "roc_curve.png"

        ),

        dpi=DPI

    )

    plt.close()

    print("ROC Curve Saved.")
    
    
# Precision-Recall Curve generate garne function

def plot_precision_recall_curve(
        y_true,
        y_probability):
    """
    Precision Recall Curve
    """

    precision, recall, thresholds = precision_recall_curve(

        y_true,

        y_probability

    )

    # CSV save gareko
    pr_df = pd.DataFrame({

        "Precision": precision[:-1],

        "Recall": recall[:-1],

        "Threshold": thresholds

    })

    pr_df.to_csv(

        os.path.join(

            CSV_DIR,

            "pr_curve_values.csv"

        ),

        index=False

    )

    plt.figure(figsize=FIG_SIZE)

    plt.plot(

        recall,

        precision,

        linewidth=2

    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision Recall Curve")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOT_DIR,

            "precision_recall_curve.png"

        ),

        dpi=DPI

    )

    plt.close()

    print("Precision Recall Curve Saved.")
    
    
def save_feature_importance_csv(
        model,
        feature_names):
    """
    Feature Importance CSV save garne function
    """

    # XGBoost / Random Forest
    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

    # Linear SVM
    elif hasattr(model, "coef_"):

        importance = np.abs(model.coef_[0])

    # Other models
    else:

        print("Feature importance is not available for this model.")

        return None

    # DataFrame banayeko
    feature_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance

    })

    # Importance descending order ma sort gareko
    feature_df = feature_df.sort_values(

        by="Importance",

        ascending=False

    )

    # CSV save gareko
    feature_df.to_csv(

        os.path.join(

            CSV_DIR,

            "feature_importance.csv"

        ),

        index=False

    )

    print("Feature Importance CSV Saved.")

    return feature_df
  
  
def plot_feature_importance(
        feature_df):
    """
    Top 20 Feature Importance
    """

    # Top 20 features matra select gareko
    top_features = feature_df.head(20)

    plt.figure(figsize=(12,8))

    # Horizontal bar plot banayeko
    plt.barh(

        top_features["Feature"],

        top_features["Importance"]

    )

    plt.title("Top 20 Feature Importance")

    plt.xlabel("Importance Score")

    plt.ylabel("Feature")

    plt.grid(True)

    # Highest importance mathi dekhuna reverse gareko
    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOT_DIR,

            "feature_importance.png"

        ),

        dpi=DPI

    )

    plt.close()

    print("Top Feature Importance Saved.")
    
    
def plot_complete_feature_importance(
        feature_df):
    """
    Complete Feature Importance Plot
    """

    plt.figure(figsize=(14,20))

    plt.barh(

        feature_df["Feature"],

        feature_df["Importance"]

    )

    plt.title("Complete Feature Importance")

    plt.xlabel("Importance Score")

    plt.ylabel("Feature")

    plt.grid(True)

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOT_DIR,

            "complete_feature_importance.png"

        ),

        dpi=DPI

    )

    plt.close()

    print("Complete Feature Importance Saved.")
    
    
def plot_prediction_distribution(y_pred):
    """
    Prediction Distribution Plot
    """

    # Prediction count calculate gareko
    prediction_count = pd.Series(
        y_pred
    ).value_counts().sort_index()

    plt.figure(figsize=FIG_SIZE)

    # Bar chart banayeko
    plt.bar(

        ["Bonafide", "Spoof"],

        prediction_count.values

    )

    plt.title("Prediction Distribution")

    plt.xlabel("Predicted Class")

    plt.ylabel("Count")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOT_DIR,

            "prediction_distribution.png"

        ),

        dpi=DPI

    )

    plt.close()

    print("Prediction Distribution Saved.")
    

def plot_probability_distribution(
        y_probability):
    """
    Probability Distribution
    """

    plt.figure(figsize=FIG_SIZE)

    plt.hist(

        y_probability,

        bins=30

    )

    plt.title("Prediction Probability Distribution")

    plt.xlabel("Spoof Probability")

    plt.ylabel("Frequency")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOT_DIR,

            "probability_distribution.png"

        ),

        dpi=DPI

    )

    plt.close()

    print("Probability Distribution Saved.")
    

def plot_label_distribution(
        labels):
    """
    Label Distribution
    """

    label_count = pd.Series(

        labels

    ).value_counts().sort_index()

    plt.figure(figsize=FIG_SIZE)

    plt.bar(

        ["Bonafide", "Spoof"],

        label_count.values

    )

    plt.title("Label Distribution")

    plt.xlabel("Class")

    plt.ylabel("Count")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            PLOT_DIR,

            "label_distribution.png"

        ),

        dpi=DPI

    )

    plt.close()

    print("Label Distribution Saved.")
    
    
def plot_correlation_heatmap(
        dataframe,
        feature_names=None):
    """
    Correlation Heatmap
    """

    # Numpy array aaye DataFrame banaune
    if isinstance(dataframe, np.ndarray):

        dataframe = pd.DataFrame(
            dataframe,
            columns=feature_names
        )

    correlation = dataframe.corr()

    plt.figure(figsize=(18, 16))

    sns.heatmap(
        correlation,
        cmap="coolwarm"
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "correlation_heatmap.png"
        ),
        dpi=DPI
    )

    plt.close()

    print("Correlation Heatmap Saved.")