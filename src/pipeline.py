"""
====================================================
Main Pipeline
====================================================

This is the main entry point of the project.

Pipeline Flow:

1. Load Dataset
2. Explore Dataset
3. Prepare Features
4. Encode Labels
5. Use Official Train/Dev Split
6. Scale Features (Optional)
7. Ready for Model Training
"""

# ==========================================
# Import Configuration
# ==========================================

from config import *

# ==========================================
# Import Utility Functions
# ==========================================

from utils import *

# ==========================================
# Import Dataset Loader
# ==========================================

from data_loader import load_datasets

# ==========================================
# Import Dataset Exploration
# ==========================================

from exploration import explore_dataset

# ==========================================
# Import Preprocessing Functions
# ==========================================

from preprocessing import prepare_features
from preprocessing import encode_labels
from preprocessing import scale_features


# Training Function
from trainer import train_xgboost


# Evaluator Function 
from evaluator import evaluate_model



# Visualizer Function
from visualizer import plot_confusion_matrix

from visualizer import plot_roc_curve

from visualizer import plot_precision_recall_curve

from visualizer import save_feature_importance_csv

from visualizer import plot_feature_importance

from visualizer import plot_complete_feature_importance

from visualizer import plot_prediction_distribution

from visualizer import plot_probability_distribution

from visualizer import plot_label_distribution

from visualizer import plot_correlation_heatmap


# Saver Function
from saver import save_model

from saver import save_label_encoder

def main():

    # ==========================================
    # Pipeline Start
    # ==========================================

    print_title("Audio Spoof Detection Pipeline")

    # ==========================================
    # Load Dataset
    # ==========================================

    print_status("Loading Dataset...")

    # Training ra Development dataset load gareko
    train_df, dev_df = load_datasets()

    print_status("Dataset Loaded Successfully")

    # ==========================================
    # Dataset Exploration
    # ==========================================

    print_status("Exploring Training Dataset...")

    # Training dataset explore gareko
    explore_dataset(train_df, "Training Dataset")

    print_status("Exploring Development Dataset...")

    # Development dataset explore gareko
    explore_dataset(dev_df, "Development Dataset")

    print_status("Exploration Completed")

    # ==========================================
    # Feature Preparation
    # ==========================================

    print_status("Preparing Features...")

    # Training dataset bata Features ra Label separate gareko
    X_train_full, y_train_full = prepare_features(train_df)

    # Feature names save gareko
    # Scaling pachi pani yo use garna milcha
    feature_names = X_train_full.columns.tolist()

    # Development dataset bata Features ra Label separate gareko
    X_dev, y_dev = prepare_features(dev_df)

    print_status("Feature Preparation Completed")

    # ==========================================
    # Label Encoding
    # ==========================================

    print_status("Encoding Labels...")

    # Training label lai fit gareko
    # Development label lai transform gareko
    y_train_full, y_dev, encoder = encode_labels(
        y_train_full,
        y_dev
    )

    print_status("Label Encoding Completed")

    # ==========================================
    # Official ASVspoof Train / Dev Split
    # ==========================================

    print_status("Preparing Official Train and Development Sets...")

    # Training dataset
    X_train = X_train_full
    y_train = y_train_full

    # Development dataset (Validation)
    X_valid = X_dev
    y_valid = y_dev

    print_status("Official Train and Development Sets Ready")

    # ==========================================
    # Feature Scaling (Optional)
    # ==========================================

    print_status("Scaling Features...")

    # XGBoost ko lagi scaling OFF cha
    # SVM ko lagi later ON garna sakincha
    X_train, X_valid, scaler = scale_features(
        X_train,
        X_valid
    )

    print_status("Scaling Completed")
    

    # ==========================================
    # Dataset Information
    # ==========================================

    print_divider()

    print("Training Feature Shape :", X_train.shape)
    print("Training Label Shape   :", y_train.shape)

    print()

    print("Validation Feature Shape :", X_valid.shape)
    print("Validation Label Shape   :", y_valid.shape)

    print_divider()

    # ==========================================
    # Pipeline Completed
    # ==========================================

    print_status("Preprocessing Completed Successfully")
    print_status("Ready for XGBoost Model Training...")
    
    # ==========================================
    # Model Training
    # ==========================================

    print_status("Training XGBoost Model...")

    # XGBoost model train gareko
    model = train_xgboost(
        X_train,
        y_train
    )

    print_status("Model Training Completed")
    
    # ==========================================
    # Model Evaluation
    # ==========================================

    print_status("Evaluating Model...")

    results = evaluate_model(

        model,

        X_valid,

        y_valid

    )

    print_status("Evaluation Completed")
    
    
    # ==========================================
    # Visualization
    # ==========================================

    print_status("Generating Visualizations...")

    # Confusion Matrix Heatmap
    plot_confusion_matrix(

        results["confusion_matrix"]

    )

    # ROC Curve
    plot_roc_curve(

        y_valid,

        results["y_prob"]

    )

    # Precision Recall Curve
    plot_precision_recall_curve(

        y_valid,

        results["y_prob"]

    )

    print_status("Visualization Completed")
    
    
    # ==========================================
    # Feature Importance
    # ==========================================

    print_status("Generating Feature Importance...")

    # Feature importance CSV save gareko
    feature_df = save_feature_importance_csv(

        model,

        feature_names

    )

    # Feature importance available cha bhane matra plot garne
    if feature_df is not None:

        # Top 20 Feature Importance
        plot_feature_importance(

            feature_df

        )

        # Complete Feature Importance
        plot_complete_feature_importance(

            feature_df

        )

    else:

        print("Feature Importance is not available for this model.")

    print_status("Feature Importance Completed")
    
    # ==========================================
    # Additional Visualizations
    # ==========================================

    print_status("Generating Additional Visualizations...")

    # Prediction distribution
    plot_prediction_distribution(

        results["y_pred"]

    )

    # Probability distribution
    plot_probability_distribution(

        results["y_prob"]

    )

    # Label distribution
    plot_label_distribution(

        y_valid

    )

    # Correlation heatmap
    plot_correlation_heatmap(

        X_train

    )

    print_status("Additional Visualizations Completed")
    
    # ==========================================
    # Save Model
    # ==========================================

    print_status("Saving Model...")

    # Trained model save gareko
    save_model(

        model

    )

    # Label Encoder save gareko
    save_label_encoder(

        encoder

    )

    print_status("Model Saved Successfully")


# ==========================================
# Main Function
# ==========================================

if __name__ == "__main__":

    # Main function call gareko
    main()