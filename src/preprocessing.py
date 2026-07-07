"""
====================================================
Preprocessing Module
====================================================

Yo module le

1. audio_id remove garcha
2. X ra y separate garcha
3. Label encode garcha
4. Feature scaling (optional)
5. Train Validation Split garcha
"""

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from config import *


#Since audio_id le Audio chai Deepfake or Genuine ho bhanera identify garna lai use hudaina so eslai Remove garincha
#also X is input Features so esma Target Class/COlumn lai pani Seperate garincha

def prepare_features(df):
    """
    Features ra Label separate garne
    """

    # audio_id remove gareko
    X = df.drop(columns=["audio_id", "label"])

    # label matra select gareko
    y = df["label"]

    return X, y
  

#validation data lai fit garidaina because Data Leakage huna sakcha so only training data lai fit garincha

#Label Encoding ko Kaam chai Label lai Numeric ma Convert garne ho jasto ki Genuine lai 0 ra Deepfake lai 1 banaucha
def encode_labels(y_train, y_dev):
    """
    Label lai numeric banako
    """

    # LabelEncoder object create gareko
    encoder = LabelEncoder()

    # Training label fit gareko
    y_train_encoded = encoder.fit_transform(y_train)

    # Dev label transform gareko
    y_dev_encoded = encoder.transform(y_dev)

    # Encoder save gareko
    joblib.dump(
        encoder,
        f"{MODEL_DIR}/label_encoder.joblib"
    )

    print("\nLabel Mapping")

    for index, label in enumerate(encoder.classes_):
        print(f"{label} -> {index}")

    return y_train_encoded, y_dev_encoded, encoder
  


# XGBoost Chai Tree Based Algorithm bhaekaale Eslai Scaling pardaina 
# But SVM Calculates Distance so Eslai chai Scaling garna parcha

def scale_features(X_train, X_dev):
    """
    Feature Scaling (Optional)
    XGBoost ko lagi required chaina.
    SVM ko lagi use garna sakincha.
    """

    # Scaling OFF cha bhane
    if not USE_SCALER:

        print("\nScaling Disabled")

        return X_train, X_dev, None

    print("\nScaling Enabled")

    scaler = StandardScaler()

    # Train ma fit gareko
    X_train = scaler.fit_transform(X_train)

    # Dev ma transform matra gareko
    X_dev = scaler.transform(X_dev)

    return X_train, X_dev, scaler
  


#Abo DataSet lai Split garne 
# Training lai 80% of Dataset and Validation lai 20% of Dataset

#But In Our Case Already 2 ta Dataset cha Training Dataset and Development Dataset so eslai Split garna pardaina

def split_dataset(X, y):
    """
    Train Validation Split
    """

    X_train, X_valid, y_train, y_valid = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    print("\nDataset Split Completed")

    print(f"Training Samples : {len(X_train)}")

    print(f"Validation Samples : {len(X_valid)}")

    return X_train, X_valid, y_train, y_valid