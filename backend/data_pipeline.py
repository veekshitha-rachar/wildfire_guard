"""
wildfire_system/backend/data_pipeline.py
----------------------------------------
Step 1: Load & clean all datasets
Step 2: Feature engineering → merged training dataframe
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# ─────────────────────────────────────────
# STEP 1  Load and clean each raw dataset
# ─────────────────────────────────────────

def load_sensor_data():
    """IoT sensor readings: temperature, humidity, smoke, air quality."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'sensor_wildfire_dataset.csv'))
    # Standardise column names to lowercase with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    # Drop rows where key environmental readings are missing
    df.dropna(subset=['temperature_c', 'humidity_percent'], inplace=True)
    # Encode time_of_day → numeric (morning=0, afternoon=1, evening=2, night=3)
    tod_map = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
    df['time_of_day_enc'] = df['time_of_day'].str.lower().map(tod_map).fillna(1)
    return df


def load_powerline_data():
    """Power-line proximity and infrastructure risk features."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'powerline_wildfire_dataset.csv'))
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.dropna(subset=['sparking_probability', 'powerline_risk_index'], inplace=True)
    return df


def load_human_activity_data():
    """Human presence, campfire reports, illegal burning data."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'human_activity_wildfire_dataset.csv'))
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.dropna(subset=['human_activity_index'], inplace=True)
    return df


# ─────────────────────────────────────────
# STEP 2  Feature Engineering & Merging
# ─────────────────────────────────────────

def engineer_features(sensor_df, power_df, human_df):
    """
    Merge the three datasets by row-index (since location IDs differ)
    and build composite risk features.
    Assumes each dataset has the same number of rows.
    """

    # Trim to the common length
    n = min(len(sensor_df), len(power_df), len(human_df))
    sensor_df  = sensor_df.iloc[:n].reset_index(drop=True)
    power_df   = power_df.iloc[:n].reset_index(drop=True)
    human_df   = human_df.iloc[:n].reset_index(drop=True)

    merged = pd.DataFrame()

    # ── Environmental / sensor features ──────────────────────────────
    merged['temperature_c']      = sensor_df['temperature_c']
    merged['humidity_percent']   = sensor_df['humidity_percent']
    merged['smoke_detected']     = sensor_df['smoke_detected']
    merged['air_quality_index']  = sensor_df['air_quality_index']
    merged['sensor_risk_index']  = sensor_df['sensor_risk_index']
    merged['time_of_day']        = sensor_df['time_of_day_enc']

    # ── Power-line risk features ──────────────────────────────────────
    merged['power_line_risk']        = power_df['powerline_risk_index']
    merged['sparking_probability']   = power_df['sparking_probability']
    merged['wind_speed_kmh']         = power_df['wind_speed_kmh']
    merged['vegetation_density']     = power_df['vegetation_density']
    merged['distance_to_powerline']  = power_df['distance_to_powerline_km']
    merged['line_age_years']         = power_df['line_age_years']

    # ── Human activity features ───────────────────────────────────────
    merged['human_activity_index']   = human_df['human_activity_index']
    merged['campfire_reports']        = human_df['campfire_reports']
    merged['illegal_burning_reports'] = human_df['illegal_burning_reports']
    merged['crowd_density']           = human_df['crowd_density']

    # ── Composite derived features ────────────────────────────────────
    # vegetation_dryness: high temp + low humidity = dry vegetation
    merged['vegetation_dryness'] = (
        merged['temperature_c'] / 50.0 +
        (100 - merged['humidity_percent']) / 100.0
    ).clip(0, 1)

    # overall_risk_score: weighted combination of the three risk indices
    merged['overall_risk_score'] = (
        0.40 * merged['sensor_risk_index'] +
        0.35 * merged['power_line_risk'] +
        0.25 * merged['human_activity_index']
    )

    # ── Target label ──────────────────────────────────────────────────
    # Use the sensor dataset's risk_category as ground truth
    label_map = {'Low': 0, 'Medium': 1, 'High': 2}
    merged['risk_label'] = sensor_df['risk_category'].map(label_map).fillna(0).astype(int)

    # ── Normalise all numeric features to [0, 1] ──────────────────────
    feature_cols = [c for c in merged.columns if c != 'risk_label']
    scaler = MinMaxScaler()
    merged[feature_cols] = scaler.fit_transform(merged[feature_cols])

    return merged, feature_cols, scaler


def build_training_data():
    """Public entry-point: returns (X, y, feature_cols, scaler)."""
    sensor_df = load_sensor_data()
    power_df  = load_powerline_data()
    human_df  = load_human_activity_data()
    merged, feature_cols, scaler = engineer_features(sensor_df, power_df, human_df)

    X = merged[feature_cols]
    y = merged['risk_label']
    return X, y, feature_cols, scaler


if __name__ == '__main__':
    X, y, cols, scaler = build_training_data()
    print(f"Training data shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    print(f"Features: {cols}")
