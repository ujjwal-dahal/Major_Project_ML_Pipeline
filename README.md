# 🎙️ Audio Spoof Detection using Machine Learning (ASVspoof5)

A modular Machine Learning pipeline for detecting spoofed audio using handcrafted acoustic features extracted from the **ASVspoof5** dataset. The project is designed with a clean and reusable architecture so that different machine learning models (XGBoost, SVM, Random Forest, etc.) can be trained and compared with minimal code changes.

---

## 📌 Project Overview

Speaker verification systems are vulnerable to spoofing attacks such as:

* Replay Attacks
* Text-to-Speech (TTS)
* Voice Conversion (VC)
* AI Generated Speech

This project detects whether an input audio sample is:

* **Bonafide** (Real Human Speech)
* **Spoof** (Fake/Artificial Speech)

using handcrafted audio features and Machine Learning.

---

## 🚀 Features

* Modular ML Pipeline
* ASVspoof5 Dataset Support
* Handcrafted Audio Feature Extraction
* XGBoost Classifier
* Easy Model Replacement (SVM, Random Forest, etc.)
* Comprehensive Evaluation Metrics
* Automatic Visualization Generation
* Automatic Model Saving
* Production-Style Project Structure

---

## 📂 Project Structure

```text
Audio-Spoof-Detection/
│
├── data/
│   ├── features_train.csv
│   └── features_dev.csv
│
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── exploration.py
│   ├── preprocessing.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── visualizer.py
│   ├── saver.py
│   └── pipeline.py
│
├── results/
│   └── XGBoost/
│       ├── models/
│       ├── csv/
│       ├── plots/
│       └── logs/
│
├── requirements.txt
└── README.md
```

---

## 🗂 Dataset

This project uses the **ASVspoof5** dataset.

The dataset is **not included** in this repository because of its large size and licensing restrictions.

After preprocessing and feature extraction, create the following files inside the `data/` directory:

```text
data/
│
├── features_train.csv
└── features_dev.csv
```

Each CSV contains:

* 74 handcrafted audio features
* audio_id
* label

---

## 🎵 Handcrafted Features

The extracted features include:

### MFCC

* Mean
* Standard Deviation

### Chroma

* Mean
* Standard Deviation

### Zero Crossing Rate (ZCR)

* Mean
* Standard Deviation

### RMS Energy

* Mean
* Standard Deviation

### Spectral Centroid

* Mean
* Standard Deviation

### Spectral Bandwidth

* Mean
* Standard Deviation

### Spectral Rolloff

* Mean
* Standard Deviation

### Spectral Contrast

* Mean
* Standard Deviation

Total Features:

**74**

---

## ⚙️ Machine Learning Pipeline

The complete pipeline consists of:

1. Dataset Loading
2. Dataset Exploration
3. Feature Preparation
4. Label Encoding
5. Optional Feature Scaling
6. Model Training
7. Prediction
8. Model Evaluation
9. Visualization
10. Model Saving

---

## 🤖 Current Model

* XGBoost

The project is designed so the model can easily be replaced with:

* SVM
* Random Forest
* LightGBM
* CatBoost
* Logistic Regression

without changing the entire pipeline.

---

## 📊 Evaluation Metrics

The pipeline computes:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Balanced Accuracy
* Specificity
* Sensitivity
* Log Loss
* Matthews Correlation Coefficient (MCC)
* Cohen's Kappa
* False Positive Rate
* False Negative Rate
* True Positive Rate
* True Negative Rate
* Confusion Matrix
* Classification Report

---

## 📈 Generated Visualizations

The pipeline automatically generates:

* Label Distribution
* Prediction Distribution
* Probability Distribution
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Top 20 Feature Importance
* Complete Feature Importance
* Correlation Heatmap

---

## 💾 Generated Outputs

```text
results/
│
├── models/
│   ├── xgboost_model.joblib
│   └── label_encoder.joblib
│
├── csv/
│   ├── evaluation_metrics.csv
│   ├── validation_predictions.csv
│   ├── feature_importance.csv
│   ├── roc_curve_values.csv
│   └── pr_curve_values.csv
│
└── plots/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── feature_importance.png
    ├── complete_feature_importance.png
    ├── prediction_distribution.png
    ├── probability_distribution.png
    ├── label_distribution.png
    └── correlation_heatmap.png
```

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
```

Move into the project directory:

```bash
cd your-repository
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python src/pipeline.py
```

---

## 📌 Future Improvements

* Deep Learning Models
* CNN-based Audio Classification
* Transformer-based Audio Models
* Wav2Vec2
* HuBERT
* Ensemble Learning
* Hyperparameter Optimization
* Cross Validation
* SHAP Explainability
* Real-time Audio Spoof Detection

---

## 👨‍💻 Author

**Ujjwal Dahal**

Bachelor in Electronics, Communication and Information Engineering

Institute of Engineering (IOE), Thapathali Campus

GitHub: https://github.com/ujjwal-dahal

---

## 📄 License

This project is intended for educational and research purposes.
