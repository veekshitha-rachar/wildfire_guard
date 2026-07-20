import os
import json
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# ===============================
# LOAD MODEL FILES
# ===============================
with open(os.path.join(MODELS_DIR, "rf_model.pkl"), "rb") as f:
    rf_model = pickle.load(f)

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODELS_DIR, "feature_cols.json")) as f:
    feature_cols = json.load(f)

print("✓ Model loaded")

# ===============================
# LOAD DATASETS
# ===============================
human = pd.read_csv(os.path.join(DATA_DIR, "human_activity_wildfire_dataset.csv"))
power = pd.read_csv(os.path.join(DATA_DIR, "powerline_wildfire_dataset.csv"))
sensor = pd.read_csv(os.path.join(DATA_DIR, "sensor_wildfire_dataset.csv"))

print("Datasets loaded successfully")
print("Human:", human.shape)
print("Power:", power.shape)
print("Sensor:", sensor.shape)

# ===============================
# COMBINE DATA
# ===============================
df = pd.concat([human, power, sensor], axis=1)

# ===============================
# HANDLE MISSING FEATURES
# ===============================
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0

# ===============================
# FIX TEXT → NUMERIC (IMPORTANT)
# ===============================
time_map = {
    "Morning": 0,
    "Afternoon": 1,
    "Evening": 2,
    "Night": 3
}

if "time_of_day" in df.columns:
    df["time_of_day"] = df["time_of_day"].map(time_map).fillna(1)

# ===============================
# PREPARE INPUT DATA
# ===============================
X = df[feature_cols]
X_scaled = scaler.transform(X)

# ===============================
# TARGET LABEL
# ===============================
if "risk_category" in sensor.columns:
    y_true_text = sensor["risk_category"]
elif "risk_category" in human.columns:
    y_true_text = human["risk_category"]
else:
    raise ValueError("❌ No risk_category column found")

label_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

y_true = y_true_text.map(label_map)

# ===============================
# PREDICTION
# ===============================
y_pred = rf_model.predict(X_scaled)

# ===============================
# EVALUATION OUTPUT
# ===============================
print("\n========== MODEL EVALUATION ==========")

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Low", "Medium", "High"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ===============================
# FEATURE IMPORTANCE
# ===============================
if hasattr(rf_model, "feature_importances_"):
    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 10 Important Features:")
    print(importance.head(10))