"""
WildFireGuard v2 — run.py
--------------------------
One-command startup:
  python run.py

Steps:
  1. Trains models if not already trained
  2. Starts Flask backend (serves both frontend + admin)

Access:
  Login      →  http://localhost:5000/login
  User App   →  http://localhost:5000         (user / user123)
  Admin      →  http://localhost:5000/admin   (admin / admin123)
  Health     →  http://localhost:5000/health
  SHAP       →  http://localhost:5000/model/explain
"""

import os
import sys

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def models_exist():
    return os.path.exists(os.path.join(MODELS_DIR, "rf_model.pkl"))


if __name__ == "__main__":
    # Step 1 — train if needed
    if not models_exist():
        print("Models not found — training now…\n")
        sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
        from train_model import train
        train()
    else:
        print("✓ Models already trained — skipping training.\n")

    # Step 2 — start Flask
    print("=" * 60)
    print("  WildFireGuard v2 starting…")
    print("  Login Page : http://localhost:5000/login")
    print("  User App   : http://localhost:5000  (user / user123)")
    print("  Admin      : http://localhost:5000/admin  (admin / admin123)")
    print("  Health     : http://localhost:5000/health")
    print("=" * 60 + "\n")

    sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
    from app import app
    app.run(debug=True, host="0.0.0.0", port=5000)
