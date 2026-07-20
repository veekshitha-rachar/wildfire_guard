"""
WildFireGuard — backend/train_voice_distress.py
================================================
OPTIONAL training script for audio-based distress detection.
This is NOT required to run the app. The app uses Web Speech API
(browser) to convert voice → text, then sends text to /distress.

This script is for FUTURE enhancement: training an audio emotion
classifier that can detect distress directly from audio features
(MFCC, pitch, energy, spectrogram) WITHOUT speech-to-text.

────────────────────────────────────────────────────────────────
DATASETS (download separately — links in README):

  RAVDESS  → https://zenodo.org/records/1188976
    - 24 actors, 8 emotions (calm, happy, sad, angry, fearful…)
    - Format: 24bit/48kHz WAV audio files
    - License: Creative Commons BY-NC-SA 4.0

  CREMA-D  → https://github.com/CheyneyComputerScience/CREMA-D
    - 7,442 clips from 91 actors, 6 emotions
    - Format: WAV files (16kHz mono)
    - License: Open Access

  TESS     → https://huggingface.co/datasets/myleslinder/tess
    - Toronto Emotional Speech Set
    - 2800 clips, 7 emotions, 2 female actors
    - Format: WAV files
    - License: Creative Commons BY-NC-SA

────────────────────────────────────────────────────────────────
PREREQUISITES (uncomment in requirements.txt before running):

  pip install librosa soundfile scikit-learn numpy

────────────────────────────────────────────────────────────────
PIPELINE:
  audio file
    → load with librosa (16kHz mono)
    → extract features:
        MFCC (40 coefficients)
        Delta MFCC
        Chroma
        Mel Spectrogram statistics
        Pitch (F0) mean & std
        Energy (RMS)
        Zero Crossing Rate
    → concatenate into feature vector (~200 dims)
    → train Random Forest / SVM classifier
    → binary label: distress (fearful/angry/sad) vs normal

────────────────────────────────────────────────────────────────
Run (after downloading a dataset):
  python backend/train_voice_distress.py --dataset ravdess --path /path/to/dataset
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np

# ── Check for required packages ──────────────────────────────────────
try:
    import librosa
    import soundfile as sf
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    print("✓ All packages available")
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("   Run: pip install librosa soundfile scikit-learn")
    sys.exit(1)


# ── Distress emotion labels per dataset ──────────────────────────────
# RAVDESS emotion codes: 01=neutral,02=calm,03=happy,04=sad,
#                        05=angry,06=fearful,07=disgust,08=surprised
RAVDESS_DISTRESS_CODES = {"05", "06", "07"}  # angry, fearful, disgust

# CREMA-D emotion codes: ANG,DIS,FEA,HAP,NEU,SAD
CREMAD_DISTRESS_CODES  = {"ANG", "DIS", "FEA", "SAD"}

# TESS emotion labels (folder names)
TESS_DISTRESS_LABELS   = {"angry", "disgust", "fear", "sad"}


# ── Feature extraction ───────────────────────────────────────────────

def extract_features(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Load an audio file and extract a fixed-size feature vector.

    Features extracted:
      - MFCC (40 coefficients → mean + std = 80 values)
      - Delta MFCC (80 values)
      - Chroma (12 → mean + std = 24 values)
      - Mel Spectrogram (128 bands → mean + std = 256 values)
      - RMS Energy (mean + std = 2 values)
      - Zero Crossing Rate (mean + std = 2 values)
      - Pitch / F0 (mean + std = 2 values)

    Total: ~446 dimensions
    """
    try:
        y, _sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        print(f"   ⚠ Cannot load {audio_path}: {e}")
        return None

    feats = []

    # MFCC
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc,  axis=1))

    # Delta MFCC
    delta = librosa.feature.delta(mfcc)
    feats.extend(np.mean(delta, axis=1))
    feats.extend(np.std(delta,  axis=1))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(np.mean(chroma, axis=1))
    feats.extend(np.std(chroma,  axis=1))

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats.extend(np.mean(mel_db, axis=1))
    feats.extend(np.std(mel_db,  axis=1))

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    feats.append(float(np.mean(rms)))
    feats.append(float(np.std(rms)))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    feats.append(float(np.mean(zcr)))
    feats.append(float(np.std(zcr)))

    # Pitch (Fundamental Frequency / F0)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"),
                             fmax=librosa.note_to_hz("C7"))
    f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
    feats.append(float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0)
    feats.append(float(np.std(f0_clean))  if len(f0_clean) > 0 else 0.0)

    return np.array(feats, dtype=np.float32)


# ── Dataset loaders ──────────────────────────────────────────────────

def load_ravdess(dataset_path: str):
    """Load RAVDESS dataset and extract features."""
    X, y = [], []
    for actor_folder in sorted(os.listdir(dataset_path)):
        actor_path = os.path.join(dataset_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        for fname in sorted(os.listdir(actor_path)):
            if not fname.endswith(".wav"):
                continue
            parts = fname.replace(".wav","").split("-")
            if len(parts) < 3:
                continue
            emotion_code = parts[2]
            label = 1 if emotion_code in RAVDESS_DISTRESS_CODES else 0
            feats = extract_features(os.path.join(actor_path, fname))
            if feats is not None:
                X.append(feats); y.append(label)
    return np.array(X), np.array(y)


def load_cremad(dataset_path: str):
    """Load CREMA-D dataset."""
    X, y = [], []
    audio_dir = os.path.join(dataset_path, "AudioWAV")
    if not os.path.exists(audio_dir):
        audio_dir = dataset_path
    for fname in sorted(os.listdir(audio_dir)):
        if not fname.endswith(".wav"):
            continue
        emotion_code = fname.split("_")[2].upper() if "_" in fname else ""
        label = 1 if emotion_code in CREMAD_DISTRESS_CODES else 0
        feats = extract_features(os.path.join(audio_dir, fname))
        if feats is not None:
            X.append(feats); y.append(label)
    return np.array(X), np.array(y)


def load_tess(dataset_path: str):
    """Load TESS dataset (organized in emotion subfolders)."""
    X, y = [], []
    for emotion_folder in sorted(os.listdir(dataset_path)):
        emotion_path = os.path.join(dataset_path, emotion_folder)
        if not os.path.isdir(emotion_path):
            continue
        emotion = emotion_folder.lower()
        label = 1 if any(e in emotion for e in TESS_DISTRESS_LABELS) else 0
        for fname in sorted(os.listdir(emotion_path)):
            if not fname.endswith(".wav"):
                continue
            feats = extract_features(os.path.join(emotion_path, fname))
            if feats is not None:
                X.append(feats); y.append(label)
    return np.array(X), np.array(y)


# ── Training ──────────────────────────────────────────────────────────

def train(dataset: str, dataset_path: str, output_dir: str):
    print(f"\n🎙 Training voice distress model on {dataset.upper()} dataset")
    print(f"   Dataset path: {dataset_path}\n")

    print("Extracting audio features…")
    if dataset == "ravdess":
        X, y = load_ravdess(dataset_path)
    elif dataset == "cremad":
        X, y = load_cremad(dataset_path)
    elif dataset == "tess":
        X, y = load_tess(dataset_path)
    else:
        print(f"Unknown dataset: {dataset}. Use ravdess / cremad / tess")
        return

    print(f"✓ Extracted {len(X)} samples | Distress: {y.sum()} | Normal: {(y==0).sum()}")

    if len(X) < 10:
        print("❌ Too few samples — check your dataset path.")
        return

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split
    Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.2,
                                            random_state=42, stratify=y)

    # Train Random Forest
    print("\nTraining Random Forest classifier…")
    clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                  random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)

    # Evaluate
    ypred = clf.predict(Xte)
    print("\n── Evaluation Report ──")
    print(classification_report(yte, ypred, target_names=["Normal","Distress"]))

    # Save model + scaler
    os.makedirs(output_dir, exist_ok=True)
    model_path  = os.path.join(output_dir, "voice_distress_model.pkl")
    scaler_path = os.path.join(output_dir, "voice_distress_scaler.pkl")
    meta_path   = os.path.join(output_dir, "voice_distress_meta.json")

    with open(model_path,  "wb") as f: pickle.dump(clf,    f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
    with open(meta_path,   "w")  as f:
        json.dump({"dataset": dataset, "n_samples": len(X),
                   "n_features": X.shape[1],
                   "distress_codes": list(RAVDESS_DISTRESS_CODES)}, f, indent=2)

    print(f"\n✅ Model saved → {model_path}")
    print(f"✅ Scaler saved → {scaler_path}")
    print(f"✅ Meta saved  → {meta_path}")
    print("\nTo use this model, load it in app.py as an optional audio classifier.")


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WildFireGuard voice distress model")
    parser.add_argument("--dataset", choices=["ravdess","cremad","tess"], required=True,
                        help="Which dataset to use")
    parser.add_argument("--path",    required=True, help="Path to the downloaded dataset folder")
    parser.add_argument("--output",  default="models", help="Output dir for saved model (default: models/)")
    args = parser.parse_args()

    train(dataset=args.dataset, dataset_path=args.path, output_dir=args.output)
