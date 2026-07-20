"""
wildfire_system/backend/train_model.py
--------------------------------------
Step 3: Train Random Forest + XGBoost ensemble wildfire risk classifier.
Saves model artifacts to ../models/ as pickle files.
"""

import os
import sys
import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add parent dir so we can import data_pipeline
sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import build_training_data

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not found — using Random Forest only.")

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

RISK_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}


def train():
    print("=" * 55)
    print("  WILDFIRE RISK MODEL — TRAINING")
    print("=" * 55)

    # ── Load engineered data ────────────────────────────────────────
    X, y, feature_cols, scaler = build_training_data()
    print(f"\n✓ Dataset loaded  →  {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class distribution: {dict(y.value_counts().sort_index())}")

    # ── Train / test split ──────────────────────────────────────────
    # stratify preserves class balance in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ── Model 1 : Random Forest ─────────────────────────────────────
    print("\n[1/2] Training Random Forest …")
    rf_model = RandomForestClassifier(
        n_estimators=200,      # 200 trees for stable predictions
        max_depth=None,        # let trees grow fully
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc   = accuracy_score(y_test, rf_preds)
    print(f"   Random Forest Accuracy: {rf_acc:.4f}")

    # ── Model 2 : XGBoost ──────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        print("\n[2/2] Training XGBoost …")
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        xgb_acc   = accuracy_score(y_test, xgb_preds)
        print(f"   XGBoost Accuracy     : {xgb_acc:.4f}")
    else:
        xgb_model = None

    # ── Ensemble : average class probabilities ─────────────────────
    print("\n[Ensemble] Averaging RF + XGBoost probabilities …")
    rf_proba = rf_model.predict_proba(X_test)
    if xgb_model is not None:
        xgb_proba = xgb_model.predict_proba(X_test)
        ensemble_proba = (rf_proba + xgb_proba) / 2.0
    else:
        ensemble_proba = rf_proba

    ensemble_preds = np.argmax(ensemble_proba, axis=1)
    ens_acc = accuracy_score(y_test, ensemble_preds)
    print(f"   Ensemble Accuracy    : {ens_acc:.4f}")

    print("\n── Classification Report (Ensemble) ──")
    print(classification_report(y_test, ensemble_preds,
                                 target_names=['Low', 'Medium', 'High']))

    # ── Save artefacts ─────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)

    if xgb_model is not None:
        with open(os.path.join(MODELS_DIR, 'xgb_model.pkl'), 'wb') as f:
            pickle.dump(xgb_model, f)

    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(MODELS_DIR, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)

    meta = {
        'rf_accuracy':       round(rf_acc, 4),
        'ensemble_accuracy': round(ens_acc, 4),
        'xgboost_available': XGBOOST_AVAILABLE,
        'feature_cols':      feature_cols,
        'risk_labels':       RISK_LABELS
    }
    with open(os.path.join(MODELS_DIR, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n✓ Models saved to models/")
    print(f"  rf_model.pkl  |  scaler.pkl  |  feature_cols.json")
    return meta


if __name__ == '__main__':
    train()
