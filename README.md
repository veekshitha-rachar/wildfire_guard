# 🔥 WildFireGuard — Complete Wildfire Prediction & Distress Detection System

A full-stack prototype for wildfire risk prediction, text-based distress detection,
and GPS emergency alerting with a user app + admin dashboard.

---

## 📁 Project Structure

```
wildfire_system/
├── run.py                        ← One-command startup
├── requirements.txt
│
├── data/
│   ├── sensor_wildfire_dataset.csv
│   ├── powerline_wildfire_dataset.csv
│   └── human_activity_wildfire_dataset.csv
│
├── backend/
│   ├── data_pipeline.py          ← Step 1 & 2: Load, clean, merge, engineer features
│   ├── train_model.py            ← Step 3: Train RF + XGBoost ensemble
│   └── app.py                    ← Step 4 & 5: Flask API (predict, distress, alert)
│
├── frontend/
│   ├── index.html                ← Step 6 & 7: User app with GPS emergency button
│   └── admin.html                ← Step 8: Admin dashboard for authorities
│
└── models/                       ← Auto-created after training
    ├── rf_model.pkl
    ├── xgb_model.pkl
    ├── scaler.pkl
    ├── feature_cols.json
    └── model_meta.json
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run everything (trains + starts server)
```bash
python3 run.py
```

### 3. Open in browser
- **User App** → http://localhost:5000
- **Admin Dashboard** → http://localhost:5000/admin

---

## 🔌 API Reference

### POST /predict — Wildfire Risk
```json
Request:  { "temperature_c": 45, "humidity_percent": 15, "smoke_detected": 1, ... }
Response: { "risk_level": "High", "confidence": 92.5,
            "probabilities": { "Low": 2.1, "Medium": 5.4, "High": 92.5 } }
```

### POST /distress — Text Distress Detection
```json
Request:  { "message": "fire spreading near our house please help" }
Response: { "is_distress": true, "label": "DISTRESS", "confidence": 97.2,
            "message_processed": "fire spreading near house please help" }
```

### POST /alert — Emergency GPS Alert
```json
Request:  { "latitude": 12.9716, "longitude": 77.5946,
            "message": "Fire visible from my house", "risk_level": "High" }
Response: { "success": true, "alert_id": "A3B9C1D2", "alert": {...} }
```

### GET /alerts — List All Alerts (Admin)
```json
Response: { "alerts": [...], "total": 5 }
```

### POST /alert/respond — Respond to Alert (Admin)
```json
Request:  { "alert_id": "A3B9C1D2", "responder_note": "Team dispatched" }
Response: { "success": true, "alert": { "status": "RESPONDED", ... } }
```

---

## 🤖 ML Models

| Component         | Algorithm               | Input Features         |
|-------------------|-------------------------|------------------------|
| Wildfire Risk     | RF + XGBoost Ensemble   | 18 sensor/env features |
| Distress Detection| TF-IDF + Logistic Reg.  | Tweet/message text     |

**Risk Levels:** Low → Medium → High  
**Distress Labels:** NORMAL / DISTRESS

---

## ☁️ Cloud Deployment

### Backend → Render / Railway
1. Push project to GitHub
2. Connect to Render (render.com) → New Web Service
3. Set **Start Command**: `gunicorn wildfire_system.backend.app:app`
4. Set **Root Directory**: `wildfire_system`

### Frontend → Same server (Flask serves static files)
The Flask app serves `frontend/index.html` and `frontend/admin.html`
directly — no separate hosting needed.

### Alternative: Streamlit UI
```bash
pip install streamlit
streamlit run streamlit_app.py   # (build on top of backend APIs)
```

---

## 📊 Datasets Used

| Dataset              | Source         | Features Used                            |
|----------------------|----------------|------------------------------------------|
| sensor_wildfire      | Custom (yours) | temperature, humidity, smoke, AQI        |
| powerline_wildfire   | Custom (yours) | sparking prob, line age, wind, vegetation|
| human_activity       | Custom (yours) | campfire reports, crowd density          |
| Forest Fires (UCI)   | Kaggle         | Extend with real fire perimeter data     |
| Disaster Tweets      | Kaggle         | Seed for distress model (extend)         |
| MODIS Fire Data      | NASA FIRMS     | Real-time fire hotspot validation        |

---

## 🔮 Extending the System

1. **More training data**: Add Kaggle datasets to `data/` and update `data_pipeline.py`
2. **Real database**: Replace `ALERTS = {}` in `app.py` with SQLite / PostgreSQL
3. **SMS alerts**: Integrate Twilio API in the `/alert` route
4. **Image CNN**: Add MobileNet inference in `/image-predict` route
5. **Real-time map**: Embed Leaflet.js map in admin dashboard with live alert pins
6. **Push notifications**: Use Firebase FCM for mobile alerts

---

*Built with Flask · scikit-learn · XGBoost · Vanilla JS · Dark UI*
