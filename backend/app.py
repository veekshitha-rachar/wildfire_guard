"""
wildfire_system/backend/app.py
------------------------------
10/10 Winning Backend Logic for Wildfire + Distress Emergency System
UPGRADED: SQLite database added for persistent alerts + email alert simulation.

Routes:
  GET    /                 -> user dashboard
  GET    /admin            -> admin dashboard
  GET    /health           -> backend health check
  POST   /predict          -> wildfire prediction + explainable AI + decision + auto alert
  POST   /distress         -> text distress detection + auto alert
  POST   /decision         -> unified emergency decision engine
  POST   /alert            -> manual emergency GPS alert
  GET    /alerts           -> admin dashboard alert list from database
  GET    /alerts/summary   -> alert analytics summary from database
  POST   /alert/respond    -> admin marks alert as responded
  POST   /alert/resolve    -> admin marks alert as resolved
  DELETE /alerts/clear     -> clear alerts for demo reset
  GET    /demo/high-risk   -> creates demo high-risk prediction + alert
"""

import os
import json
import pickle
import uuid
import re
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify, send_from_directory, redirect, session
from flask_cors import CORS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Optional SHAP explainability. App still works if SHAP is not installed.
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False



# ══════════════════════════════════════════════════════════════════════
# PATH SETUP
# ══════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
STATIC_DIR = os.path.join(ROOT_DIR, "static")
DB_PATH = os.path.join(BASE_DIR, "alerts.db")

# ══════════════════════════════════════════════════════════════════════
# EMAIL ALERT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
# Keep EMAIL_ENABLED = False for demo. It will print the email in terminal.
# To send real Gmail later:
# 1. Put your Gmail in SENDER_EMAIL
# 2. Put Gmail App Password in SENDER_PASSWORD
# 3. Put receiver email in RECEIVER_EMAIL
# 4. Change EMAIL_ENABLED = True
EMAIL_ENABLED = False

SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_gmail_app_password"
RECEIVER_EMAIL = "receiver_email@gmail.com"

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/static"
)
CORS(app)
app.secret_key = os.environ.get("WILDFIRE_SECRET_KEY", "wildfireguard-demo-secret-key")



# ══════════════════════════════════════════════════════════════════════
# SQLITE DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════

def init_db():
    """Create alerts table if it does not already exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        alert_id TEXT PRIMARY KEY,
        alert_type TEXT,
        latitude REAL,
        longitude REAL,
        message TEXT,
        risk_level TEXT,
        priority TEXT,
        source TEXT,
        explanations TEXT,
        recommended_action TEXT,
        timestamp TEXT,
        status TEXT,
        prediction_confidence REAL,
        emergency_score REAL,
        distress_probability REAL,
        timeline TEXT,
        responder_note TEXT,
        resolution_note TEXT,
        responded_at TEXT,
        resolved_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        latitude REAL,
        longitude REAL,
        temperature REAL,
        humidity REAL,
        wind REAL,
        aqi REAL,
        smoke REAL,
        human_activity_index REAL,
        power_line_risk REAL,
        risk_level TEXT,
        confidence REAL,
        emergency_score REAL
    )
    """)

    conn.commit()
    conn.close()


def db_insert_alert(alert):
    """Save a newly created alert into SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR REPLACE INTO alerts (
        alert_id, alert_type, latitude, longitude, message, risk_level,
        priority, source, explanations, recommended_action, timestamp,
        status, prediction_confidence, emergency_score, distress_probability,
        timeline, responder_note, resolution_note, responded_at, resolved_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        alert.get("alert_id"),
        alert.get("alert_type"),
        alert.get("latitude"),
        alert.get("longitude"),
        alert.get("message"),
        alert.get("risk_level"),
        alert.get("priority"),
        alert.get("source"),
        json.dumps(alert.get("explanations", [])),
        alert.get("recommended_action"),
        alert.get("timestamp"),
        alert.get("status"),
        alert.get("prediction_confidence"),
        alert.get("emergency_score"),
        alert.get("distress_probability"),
        json.dumps(alert.get("timeline", [])),
        alert.get("responder_note"),
        alert.get("resolution_note"),
        alert.get("responded_at"),
        alert.get("resolved_at")
    ))

    conn.commit()
    conn.close()


def db_fetch_alerts():
    """Fetch all alerts from SQLite as list of dictionaries."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()

    alerts = []
    for row in rows:
        alert = dict(row)

        try:
            alert["explanations"] = json.loads(alert.get("explanations") or "[]")
        except json.JSONDecodeError:
            alert["explanations"] = []

        try:
            alert["timeline"] = json.loads(alert.get("timeline") or "[]")
        except json.JSONDecodeError:
            alert["timeline"] = []

        alerts.append(alert)

    return alerts


def db_get_alert(alert_id):
    """Fetch one alert from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    alert = dict(row)

    try:
        alert["explanations"] = json.loads(alert.get("explanations") or "[]")
    except json.JSONDecodeError:
        alert["explanations"] = []

    try:
        alert["timeline"] = json.loads(alert.get("timeline") or "[]")
    except json.JSONDecodeError:
        alert["timeline"] = []

    return alert


def db_update_alert(alert):
    """Update full alert row in SQLite."""
    db_insert_alert(alert)


def db_clear_alerts():
    """Delete all alerts from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()


def db_insert_prediction_history(result):
    """Store prediction input + output for admin visualization."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    inp = result.get("input_summary", {})
    loc = result.get("location", {})
    decision = result.get("decision", {})
    cursor.execute("""
    INSERT INTO prediction_history (
        timestamp, latitude, longitude, temperature, humidity, wind, aqi, smoke,
        human_activity_index, power_line_risk, risk_level, confidence, emergency_score
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_utc(),
        loc.get("latitude"),
        loc.get("longitude"),
        inp.get("temperature_c"),
        inp.get("humidity_percent"),
        inp.get("wind_speed_kmh"),
        inp.get("air_quality_index"),
        inp.get("smoke_detected"),
        inp.get("human_activity_index"),
        inp.get("power_line_risk"),
        result.get("risk_level"),
        result.get("confidence"),
        decision.get("emergency_score")
    ))
    conn.commit()
    conn.close()


def db_fetch_prediction_history(limit=20):
    """Fetch latest prediction history rows for admin dashboard."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?", (int(limit),))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


init_db()


# ══════════════════════════════════════════════════════════════════════
# LOAD WILDFIRE PREDICTION MODEL
# ══════════════════════════════════════════════════════════════════════

def load_wildfire_model():
    with open(os.path.join(MODELS_DIR, "rf_model.pkl"), "rb") as f:
        rf_model = pickle.load(f)

    xgb_model = None
    xgb_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    return rf_model, xgb_model, scaler, feature_cols


rf_model, xgb_model, scaler, FEATURE_COLS = load_wildfire_model()
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}

print(f"✓ Wildfire model loaded — {len(FEATURE_COLS)} features")


# ══════════════════════════════════════════════════════════════════════
# DISTRESS DETECTION MODEL
# ══════════════════════════════════════════════════════════════════════

DISTRESS_SAMPLES = [
    ("fire spreading quickly help us", 1),
    ("wildfire near our house please send help", 1),
    ("evacuating now fire all around", 1),
    ("we are trapped smoke everywhere", 1),
    ("forest fire emergency please help", 1),
    ("flames reaching our street rescue needed", 1),
    ("fire burning out of control SOS", 1),
    ("smoke in house need emergency services", 1),
    ("family trapped in wildfire area", 1),
    ("wildfire has surrounded our village", 1),
    ("people injured in fire accident", 1),
    ("emergency wildfire spreading fast", 1),
    ("trapped on hilltop fire below", 1),
    ("children in danger fire nearby", 1),
    ("can't breathe heavy smoke please come", 1),
    ("urgent evacuation needed smoke everywhere", 1),
    ("need ambulance people stuck near fire", 1),
    ("please rescue us fire is close", 1),
    ("wildfire smoke making it hard to breathe", 1),
    ("burning forest near my location send rescue", 1),

    ("beautiful sunset today", 0),
    ("going for a hike this weekend", 0),
    ("weather looks nice outside", 0),
    ("having breakfast at home", 0),
    ("traffic is slow this morning", 0),
    ("watched a great movie last night", 0),
    ("cooking dinner for the family", 0),
    ("the park is lovely today", 0),
    ("meeting friends at the coffee shop", 0),
    ("reading a book by the fireplace", 0),
    ("planning a camping trip", 0),
    ("new restaurant opened downtown", 0),
    ("morning jog completed", 0),
    ("garden looks wonderful this spring", 0),
    ("studying for exams", 0),
    ("camping photos came out nice", 0),
    ("the sky is cloudy today", 0),
    ("went for a walk near the lake", 0),
]

texts, labels = zip(*DISTRESS_SAMPLES)

distress_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        C=1.0,
        max_iter=500,
        random_state=42
    ))
])

distress_pipeline.fit(texts, labels)
print("✓ Distress detection model trained")


# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def now_utc():
    return datetime.utcnow().isoformat() + "Z"


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_default_features():
    return {
        "temperature_c": 30.0,
        "humidity_percent": 50.0,
        "smoke_detected": 0.0,
        "air_quality_index": 75.0,
        "sensor_risk_index": 0.30,
        "time_of_day": 1.0,
        "power_line_risk": 0.30,
        "sparking_probability": 0.20,
        "wind_speed_kmh": 20.0,
        "vegetation_density": 0.50,
        "distance_to_powerline": 2.0,
        "distance_to_powerline_km": 2.0,
        "line_age_years": 10.0,
        "human_activity_index": 0.30,
        "campfire_reports": 0.0,
        "illegal_burning_reports": 0.0,
        "crowd_density": 10.0,
        "vegetation_dryness": 0.40,
        "overall_risk_score": 0.30,
        "rainfall_mm": 5.0,
        "soil_moisture": 0.50,
        "slope": 10.0,
        "elevation": 500.0,
        "ndvi": 0.50,
    }


def normalize_input_data(data):
    defaults = get_default_features()
    normalized = {}

    for key, default_val in defaults.items():
        normalized[key] = safe_float(data.get(key, default_val), default_val)

    normalized["location_name"] = data.get("location_name", "Unknown Area")
    normalized["latitude"] = data.get("latitude", data.get("lat", 12.9716))
    normalized["longitude"] = data.get("longitude", data.get("lng", 77.5946))

    return normalized


def generate_explanation(data, risk_level):
    reasons = []

    temperature = safe_float(data.get("temperature_c"), 30)
    humidity = safe_float(data.get("humidity_percent"), 50)
    wind = safe_float(data.get("wind_speed_kmh"), 20)
    aqi = safe_float(data.get("air_quality_index"), 75)
    smoke = safe_float(data.get("smoke_detected"), 0)
    human_activity = safe_float(data.get("human_activity_index"), 0.3)
    power_line = safe_float(data.get("power_line_risk"), 0.3)
    vegetation = safe_float(data.get("vegetation_density"), 0.5)
    dryness = safe_float(data.get("vegetation_dryness"), 0.4)
    campfire = safe_float(data.get("campfire_reports"), 0)
    illegal_burning = safe_float(data.get("illegal_burning_reports"), 0)
    sensor_risk = safe_float(data.get("sensor_risk_index"), 0.3)
    sparking = safe_float(data.get("sparking_probability"), 0.2)

    if temperature >= 35:
        reasons.append("High temperature increases the chance of fire ignition.")
    elif temperature >= 32:
        reasons.append("Temperature is moderately high, which can increase dryness.")

    if humidity <= 35:
        reasons.append("Low humidity makes vegetation dry and highly flammable.")
    elif humidity <= 45:
        reasons.append("Humidity is below normal, increasing fire sensitivity.")

    if wind >= 40:
        reasons.append("Strong wind can spread fire rapidly once ignition happens.")
    elif wind >= 30:
        reasons.append("Moderate wind may support faster fire spread.")

    if aqi >= 150:
        reasons.append("High AQI may indicate smoke, dust, or poor air conditions.")
    elif aqi >= 100:
        reasons.append("AQI is elevated, so air quality needs close monitoring.")

    if smoke == 1:
        reasons.append("Smoke was detected by the sensor input.")

    if human_activity >= 0.70:
        reasons.append("High human activity increases human-caused wildfire risk.")
    elif human_activity >= 0.50:
        reasons.append("Moderate human activity exists near the monitored area.")

    if power_line >= 0.70:
        reasons.append("Power line risk is high due to possible sparking or faults.")
    elif power_line >= 0.50:
        reasons.append("Power line conditions show moderate ignition risk.")

    if vegetation >= 0.75:
        reasons.append("Dense vegetation can act as fuel and increase spread intensity.")

    if dryness >= 0.70:
        reasons.append("Vegetation dryness is high, making the area more fire-prone.")
    elif dryness >= 0.50:
        reasons.append("Vegetation dryness is moderate and should be monitored.")

    if campfire >= 2:
        reasons.append("Campfire reports increase the possibility of accidental ignition.")

    if illegal_burning >= 1:
        reasons.append("Illegal burning reports increase human-caused fire risk.")

    if sensor_risk >= 0.70:
        reasons.append("Combined sensor risk index is high.")

    if sparking >= 0.70:
        reasons.append("Sparking probability is high near electrical infrastructure.")

    if not reasons:
        if risk_level == "Low":
            reasons.append("Current conditions are mostly stable with no major danger factor.")
        else:
            reasons.append("The model detected combined risk patterns from multiple inputs.")

    return reasons[:8]


def recommended_action_from_priority(priority):
    actions = {
        "CRITICAL": "Dispatch emergency response team immediately and start evacuation warning.",
        "HIGH": "Send early warning, notify forest/fire officials, and prepare response team.",
        "MEDIUM": "Monitor continuously, increase sensor observation, and keep local team ready.",
        "LOW": "Continue normal monitoring. No immediate emergency action required."
    }
    return actions.get(priority, "Continue monitoring and verify input data.")



def send_email_alert(alert):
    """
    Sends or simulates an email alert.
    Demo mode prints the email content in the VS Code terminal.
    This makes the system look real-world without requiring Gmail setup.
    """
    subject = f"🚨 WildFireGuard Alert: {alert.get('priority', 'UNKNOWN')} Priority"
    map_link = f"https://www.google.com/maps?q={alert.get('latitude')},{alert.get('longitude')}"

    explanations = alert.get("explanations", [])
    if explanations:
        explanation_text = "\n".join([f"- {item}" for item in explanations])
    else:
        explanation_text = "No explanation available."

    body = f"""
WildFireGuard Emergency Alert

Alert ID: {alert.get('alert_id')}
Type: {alert.get('alert_type')}
Risk Level: {alert.get('risk_level')}
Priority: {alert.get('priority')}
Status: {alert.get('status')}

Location:
Latitude: {alert.get('latitude')}
Longitude: {alert.get('longitude')}
Map Link: {map_link}

Message:
{alert.get('message')}

AI Explanation:
{explanation_text}

Emergency Score:
{alert.get('emergency_score')}

Prediction Confidence:
{alert.get('prediction_confidence')}

Recommended Action:
{alert.get('recommended_action')}

Timestamp:
{alert.get('timestamp')}
"""

    if not EMAIL_ENABLED:
        print("\n📧 EMAIL ALERT SIMULATION")
        print("To:", RECEIVER_EMAIL)
        print("Subject:", subject)
        print(body)
        print("📧 END EMAIL ALERT\n")
        return True

    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("✅ Email alert sent successfully")
        return True

    except Exception as e:
        print("❌ Email sending failed:", e)
        return False


def create_alert(alert_type, latitude, longitude, message, risk_level="Unknown",
                 priority="LOW", source="SYSTEM", explanations=None,
                 recommended_action=None, extra=None):
    alert_id = str(uuid.uuid4())[:8].upper()

    alert = {
        "alert_id": alert_id,
        "alert_type": alert_type,
        "latitude": safe_float(latitude, 12.9716),
        "longitude": safe_float(longitude, 77.5946),
        "message": message,
        "risk_level": risk_level,
        "priority": priority,
        "source": source,
        "explanations": explanations or [],
        "recommended_action": recommended_action or recommended_action_from_priority(priority),
        "timestamp": now_utc(),
        "status": "PENDING",
        "timeline": [
            {
                "time": now_utc(),
                "event": "Alert created",
                "note": f"{alert_type} alert generated by {source}"
            }
        ],
        "prediction_confidence": None,
        "emergency_score": None,
        "distress_probability": None,
        "responder_note": None,
        "resolution_note": None,
        "responded_at": None,
        "resolved_at": None
    }

    if extra:
        alert.update(extra)

    db_insert_alert(alert)
    send_email_alert(alert)

    print(f"\n🚨 ALERT CREATED [{alert_id}]")
    print(f"   Type     : {alert_type}")
    print(f"   Priority : {priority}")
    print(f"   Risk     : {risk_level}")
    print(f"   Location : {alert['latitude']}, {alert['longitude']}")
    print(f"   Message  : {message}\n")

    return alert


def unified_decision_engine(data):
    wildfire_risk = data.get("wildfire_risk", data.get("risk_level", "Low"))
    distress_detected = bool(data.get("distress_detected", False))

    power_line_risk = safe_float(data.get("power_line_risk"), 0)
    human_activity_index = safe_float(data.get("human_activity_index"), 0)
    sensor_risk_index = safe_float(data.get("sensor_risk_index"), 0)
    confidence = safe_float(data.get("confidence"), 0)

    score_breakdown = {}

    if wildfire_risk == "High":
        score_breakdown["wildfire_risk"] = 40
    elif wildfire_risk == "Medium":
        score_breakdown["wildfire_risk"] = 25
    else:
        score_breakdown["wildfire_risk"] = 10

    score_breakdown["distress_signal"] = 30 if distress_detected else 0
    score_breakdown["power_line_risk"] = round(power_line_risk * 10, 2)
    score_breakdown["human_activity_index"] = round(human_activity_index * 10, 2)
    score_breakdown["sensor_risk_index"] = round(sensor_risk_index * 10, 2)
    score_breakdown["model_confidence"] = round(min(confidence, 100) / 10, 2) if confidence > 0 else 0

    score = sum(score_breakdown.values())

    if score >= 80:
        priority = "CRITICAL"
    elif score >= 60:
        priority = "HIGH"
    elif score >= 35:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    auto_alert_required = priority in ["CRITICAL", "HIGH"] or distress_detected or wildfire_risk == "High"

    return {
        "emergency_score": round(score, 2),
        "score_breakdown": score_breakdown,
        "final_priority": priority,
        "auto_alert_required": auto_alert_required,
        "recommended_action": recommended_action_from_priority(priority)
    }



# ══════════════════════════════════════════════════════════════════════
# SHAP / FEATURE CONTRIBUTION EXPLANATION
# ══════════════════════════════════════════════════════════════════════

def readable_feature_name(feature):
    names = {
        "temperature_c": "Temperature",
        "humidity_percent": "Humidity",
        "wind_speed_kmh": "Wind Speed",
        "air_quality_index": "Air Quality Index",
        "smoke_detected": "Smoke Detected",
        "sensor_risk_index": "Sensor Risk Index",
        "power_line_risk": "Power Line Risk",
        "sparking_probability": "Sparking Probability",
        "human_activity_index": "Human Activity",
        "vegetation_density": "Vegetation Density",
        "vegetation_dryness": "Vegetation Dryness",
        "crowd_density": "Crowd Density",
        "line_age_years": "Power Line Age",
        "overall_risk_score": "Overall Risk Score",
    }
    return names.get(feature, feature.replace("_", " ").title())


def fallback_feature_contributions(raw_data, normalized):
    """Safe explanation when SHAP is unavailable. Returns top model-like contributors."""
    values = []

    def add(feature, value, reason):
        values.append({
            "feature": feature,
            "label": readable_feature_name(feature),
            "shap_value": round(float(value), 4),
            "direction": reason,
        })

    add("temperature_c", max((normalized.get("temperature_c", 30) - 25) / 60, 0), "increases risk when high")
    add("humidity_percent", max((60 - normalized.get("humidity_percent", 50)) / 60, 0), "increases risk when low")
    add("wind_speed_kmh", normalized.get("wind_speed_kmh", 0) / 100, "increases spread risk")
    add("air_quality_index", normalized.get("air_quality_index", 0) / 300, "indicates poor/smoky air")
    add("power_line_risk", normalized.get("power_line_risk", 0), "increases ignition risk")
    add("human_activity_index", normalized.get("human_activity_index", 0), "increases human-caused ignition risk")
    add("sensor_risk_index", normalized.get("sensor_risk_index", 0), "combines sensor warning signals")
    values = sorted(values, key=lambda x: abs(x["shap_value"]), reverse=True)[:5]
    return {"source": "fallback", "contributions": values}


def get_shap_explanation(feature_values, x_scaled, predicted_class, raw_data, normalized):
    """Return top 5 SHAP contributors. Falls back safely if SHAP fails."""
    if not SHAP_AVAILABLE:
        return fallback_feature_contributions(raw_data, normalized)
    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(x_scaled)
        if isinstance(shap_values, list):
            vals = shap_values[predicted_class][0]
        else:
            arr = np.array(shap_values)
            if arr.ndim == 3:
                if arr.shape[0] == 1:
                    vals = arr[0, :, predicted_class] if arr.shape[-1] >= 3 else arr[0, :]
                else:
                    vals = arr[predicted_class, 0, :]
            else:
                vals = arr[0]
        vals = np.array(vals).flatten()[:len(FEATURE_COLS)]
        top_idx = np.argsort(np.abs(vals))[::-1][:5]
        contributions = []
        for idx in top_idx:
            feature = FEATURE_COLS[int(idx)]
            value = float(vals[int(idx)])
            contributions.append({
                "feature": feature,
                "label": readable_feature_name(feature),
                "shap_value": round(value, 4),
                "direction": "increases risk" if value >= 0 else "decreases risk",
            })
        return {"source": "shap", "contributions": contributions}
    except Exception as e:
        fallback = fallback_feature_contributions(raw_data, normalized)
        fallback["error"] = str(e)[:120]
        return fallback


def predict_wildfire_from_data(raw_data):
    normalized = normalize_input_data(raw_data)

    defaults = get_default_features()
    feature_values = []

    for col in FEATURE_COLS:
        val = raw_data.get(col, normalized.get(col, defaults.get(col, 0.0)))
        feature_values.append(safe_float(val, defaults.get(col, 0.0)))

    X = np.array([feature_values])
    X_scaled = scaler.transform(X)

    rf_proba = rf_model.predict_proba(X_scaled)

    if xgb_model is not None:
        xgb_proba = xgb_model.predict_proba(X_scaled)
        proba = (rf_proba + xgb_proba) / 2.0
        model_used = "Random Forest + XGBoost Ensemble"
    else:
        proba = rf_proba
        model_used = "Random Forest"

    predicted_class = int(np.argmax(proba, axis=1)[0])
    confidence = float(np.max(proba)) * 100
    risk_label = RISK_LABELS[predicted_class]

    probabilities = {
        "Low": round(float(proba[0][0]) * 100, 1),
        "Medium": round(float(proba[0][1]) * 100, 1),
        "High": round(float(proba[0][2]) * 100, 1),
    }

    explanations = generate_explanation(normalized, risk_label)
    shap_explanation = get_shap_explanation(feature_values, X_scaled, predicted_class, raw_data, normalized)

    decision = unified_decision_engine({
        "wildfire_risk": risk_label,
        "distress_detected": bool(raw_data.get("distress_detected", False)),
        "power_line_risk": normalized.get("power_line_risk", 0),
        "human_activity_index": normalized.get("human_activity_index", 0),
        "sensor_risk_index": normalized.get("sensor_risk_index", 0),
        "confidence": confidence,
    })

    return {
        "risk_level": risk_label,
        "confidence": round(confidence, 1),
        "probabilities": probabilities,
        "model_used": model_used,
        "feature_count": len(FEATURE_COLS),
        "explanations": explanations,
        "shap_explanation": shap_explanation,
        "decision": decision,
        "recommended_action": decision["recommended_action"],
        "auto_alert_required": decision["auto_alert_required"],
        "input_summary": {
            "temperature_c": normalized["temperature_c"],
            "humidity_percent": normalized["humidity_percent"],
            "wind_speed_kmh": normalized["wind_speed_kmh"],
            "air_quality_index": normalized["air_quality_index"],
            "smoke_detected": normalized["smoke_detected"],
            "human_activity_index": normalized["human_activity_index"],
            "power_line_risk": normalized["power_line_risk"],
            "sensor_risk_index": normalized["sensor_risk_index"]
        },
        "location": {
            "location_name": normalized["location_name"],
            "latitude": safe_float(normalized["latitude"], 12.9716),
            "longitude": safe_float(normalized["longitude"], 77.5946)
        }
    }


# ══════════════════════════════════════════════════════════════════════
# FRONTEND ROUTES
# ══════════════════════════════════════════════════════════════════════

def is_logged_in(role=None):
    if not session.get("logged_in"):
        return False
    if role and session.get("role") != role:
        return False
    return True


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username == "admin" and password == "admin123":
            session["logged_in"] = True
            session["role"] = "admin"
            return redirect("/admin")
        if username == "user" and password == "user123":
            session["logged_in"] = True
            session["role"] = "user"
            return redirect("/")
        error = "Invalid username or password"

    return f"""
    <!DOCTYPE html><html><head><title>WildFireGuard Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
    *{{box-sizing:border-box;font-family:Segoe UI,Arial,sans-serif}}
    body{{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;color:white;background:linear-gradient(135deg,#100b09,#3b1008,#09251b)}}
    .card{{width:min(420px,92vw);padding:32px;border-radius:26px;background:rgba(255,255,255,.11);border:1px solid rgba(255,255,255,.18);box-shadow:0 22px 70px rgba(0,0,0,.35)}}
    h1{{margin:0 0 8px;font-size:30px}} p{{color:#f3d4c1;line-height:1.5}} label{{display:block;margin-top:14px;color:#ffe0c4;font-weight:800}}
    input{{width:100%;padding:13px;border-radius:14px;border:1px solid rgba(255,255,255,.2);background:rgba(0,0,0,.28);color:white;margin-top:7px}}
    button{{width:100%;margin-top:20px;padding:14px;border:none;border-radius:15px;background:linear-gradient(135deg,#ff8b22,#ff3131);color:white;font-weight:900;cursor:pointer}}
    .hint{{margin-top:16px;padding:12px;border-radius:14px;background:rgba(0,0,0,.25);font-size:13px;color:#ffe8d7}}
    .err{{color:#ffb0b0;font-weight:800;margin-top:10px}}
    </style></head><body><form class="card" method="POST">
    <h1>🔥 WildFireGuard Login</h1><p>Secure access for user dashboard and admin command center.</p>
    <label>Username</label><input name="username" required autofocus>
    <label>Password</label><input name="password" type="password" required>
    <button type="submit">Login</button>
    <div class="err">{error}</div>
    <div class="hint"><b>User:</b> user / user123<br><b>Admin:</b> admin / admin123</div>
    </form></body></html>
    """


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/")
def serve_user_ui():
    if not is_logged_in():
        return redirect("/login")
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/admin")
def serve_admin_ui():
    if not is_logged_in("admin"):
        return redirect("/login")
    return send_from_directory(FRONTEND_DIR, "admin.html")


# ══════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    alerts = db_fetch_alerts()
    return jsonify({
        "status": "ok",
        "backend": "running",
        "model": "loaded",
        "wildfire_features": len(FEATURE_COLS),
        "distress_model": "trained",
        "alerts": len(alerts),
        "database": "SQLite connected",
        "shap_available": SHAP_AVAILABLE,
        "version": "WildFireGuard v2 backend with SQLite, email simulation, SHAP fallback and dashboards"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}

    try:
        result = predict_wildfire_from_data(data)
        db_insert_prediction_history(result)

        if result["auto_alert_required"]:
            alert = create_alert(
                alert_type="WILDFIRE_RISK",
                latitude=result["location"]["latitude"],
                longitude=result["location"]["longitude"],
                message=f"Auto alert: {result['risk_level']} wildfire risk detected at {result['location']['location_name']}.",
                risk_level=result["risk_level"],
                priority=result["decision"]["final_priority"],
                source="PREDICTION_ENGINE",
                explanations=result["explanations"],
                recommended_action=result["recommended_action"],
                extra={
                    "prediction_confidence": result["confidence"],
                    "emergency_score": result["decision"]["emergency_score"]
                }
            )
            result["alert_created"] = True
            result["alert_id"] = alert["alert_id"]
        else:
            result["alert_created"] = False
            result["alert_id"] = None

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
            "hint": "Check whether frontend feature names match model feature_cols.json"
        }), 500


@app.route("/distress", methods=["POST"])
def distress():
    data = request.get_json(force=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    cleaned = clean_text(message)
    proba = distress_pipeline.predict_proba([cleaned])[0]
    distress_confidence = float(proba[1]) * 100
    is_distress = bool(proba[1] > 0.50)

    decision = unified_decision_engine({
        "wildfire_risk": data.get("wildfire_risk", "Low"),
        "distress_detected": is_distress,
        "power_line_risk": data.get("power_line_risk", 0.0),
        "human_activity_index": data.get("human_activity_index", 0.0),
        "sensor_risk_index": data.get("sensor_risk_index", 0.0),
        "confidence": distress_confidence
    })

    result = {
        "is_distress": is_distress,
        "label": "DISTRESS" if is_distress else "NORMAL",
        "confidence": round(max(proba) * 100, 1),
        "distress_probability": round(distress_confidence, 1),
        "message_processed": cleaned,
        "decision": decision,
        "recommended_action": decision["recommended_action"]
    }

    if is_distress:
        alert = create_alert(
            alert_type="DISTRESS_SIGNAL",
            latitude=data.get("latitude", 12.9716),
            longitude=data.get("longitude", 77.5946),
            message=message,
            risk_level=data.get("wildfire_risk", "Distress"),
            priority=decision["final_priority"],
            source="DISTRESS_DETECTOR",
            explanations=[
                "The text message contains emergency-related distress patterns.",
                "Distress detection model classified the message as urgent."
            ],
            recommended_action=decision["recommended_action"],
            extra={
                "distress_probability": round(distress_confidence, 1),
                "emergency_score": decision["emergency_score"]
            }
        )
        result["alert_created"] = True
        result["alert_id"] = alert["alert_id"]
    else:
        result["alert_created"] = False
        result["alert_id"] = None

    return jsonify(result)


@app.route("/decision", methods=["POST"])
def final_decision():
    data = request.get_json(force=True) or {}
    return jsonify(unified_decision_engine(data))


@app.route("/alert", methods=["POST"])
def send_alert():
    data = request.get_json(force=True) or {}

    lat = data.get("latitude")
    lng = data.get("longitude")

    if lat is None or lng is None:
        return jsonify({"error": "latitude and longitude are required"}), 400

    risk_level = data.get("risk_level", "Manual Alert")
    priority = data.get("priority")

    if not priority:
        priority = "HIGH" if risk_level == "High" else "MEDIUM"

    alert = create_alert(
        alert_type=data.get("alert_type", "MANUAL_EMERGENCY"),
        latitude=lat,
        longitude=lng,
        message=data.get("message", "Emergency button pressed"),
        risk_level=risk_level,
        priority=priority,
        source=data.get("source", "USER_BUTTON"),
        explanations=data.get("explanations", ["Manual emergency alert triggered by user."]),
        recommended_action=data.get("recommended_action", recommended_action_from_priority(priority))
    )

    return jsonify({
        "success": True,
        "alert_id": alert["alert_id"],
        "alert": alert
    })


@app.route("/alerts", methods=["GET"])
def list_alerts():
    alerts = db_fetch_alerts()

    status_filter = request.args.get("status")
    priority_filter = request.args.get("priority")

    if status_filter:
        alerts = [a for a in alerts if str(a.get("status", "")).upper() == status_filter.upper()]

    if priority_filter:
        alerts = [a for a in alerts if str(a.get("priority", "")).upper() == priority_filter.upper()]

    return jsonify({
        "alerts": alerts,
        "total": len(alerts)
    })


@app.route("/alerts/summary", methods=["GET"])
def alerts_summary():
    alerts = db_fetch_alerts()

    summary = {
        "total_alerts": len(alerts),
        "pending": 0,
        "responded": 0,
        "resolved": 0,
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "wildfire_alerts": 0,
        "distress_alerts": 0,
        "manual_alerts": 0
    }

    for alert in alerts:
        status = str(alert.get("status", "")).upper()
        priority = str(alert.get("priority", "")).upper()
        alert_type = str(alert.get("alert_type", "")).upper()

        if status == "PENDING":
            summary["pending"] += 1
        elif status == "RESPONDED":
            summary["responded"] += 1
        elif status == "RESOLVED":
            summary["resolved"] += 1

        if priority == "CRITICAL":
            summary["critical"] += 1
        elif priority == "HIGH":
            summary["high"] += 1
        elif priority == "MEDIUM":
            summary["medium"] += 1
        elif priority == "LOW":
            summary["low"] += 1

        if "WILDFIRE" in alert_type:
            summary["wildfire_alerts"] += 1
        elif "DISTRESS" in alert_type:
            summary["distress_alerts"] += 1
        elif "MANUAL" in alert_type:
            summary["manual_alerts"] += 1

    return jsonify(summary)


@app.route("/alert/respond", methods=["POST"])
def respond_alert():
    data = request.get_json(force=True) or {}
    alert_id = data.get("alert_id", "").upper()

    alert = db_get_alert(alert_id)
    if not alert:
        return jsonify({"error": f"Alert {alert_id} not found"}), 404

    alert["status"] = "RESPONDED"
    alert["responder_note"] = data.get("responder_note", "Emergency team dispatched")
    alert["responded_at"] = now_utc()

    timeline = alert.get("timeline", [])
    timeline.append({
        "time": now_utc(),
        "event": "Response started",
        "note": alert["responder_note"]
    })
    alert["timeline"] = timeline

    db_update_alert(alert)

    return jsonify({
        "success": True,
        "alert": alert
    })


@app.route("/alert/resolve", methods=["POST"])
def resolve_alert():
    data = request.get_json(force=True) or {}
    alert_id = data.get("alert_id", "").upper()

    alert = db_get_alert(alert_id)
    if not alert:
        return jsonify({"error": f"Alert {alert_id} not found"}), 404

    alert["status"] = "RESOLVED"
    alert["resolution_note"] = data.get("resolution_note", "Situation resolved")
    alert["resolved_at"] = now_utc()

    timeline = alert.get("timeline", [])
    timeline.append({
        "time": now_utc(),
        "event": "Alert resolved",
        "note": alert["resolution_note"]
    })
    alert["timeline"] = timeline

    db_update_alert(alert)

    return jsonify({
        "success": True,
        "alert": alert
    })


@app.route("/alerts/clear", methods=["DELETE"])
def clear_alerts():
    count = len(db_fetch_alerts())
    db_clear_alerts()
    return jsonify({
        "success": True,
        "message": f"Cleared {count} alerts from SQLite database"
    })


@app.route("/predictions/history", methods=["GET"])
def predictions_history():
    limit = request.args.get("limit", 20)
    try:
        limit = int(limit)
    except Exception:
        limit = 20
    return jsonify({"history": db_fetch_prediction_history(limit), "total": len(db_fetch_prediction_history(1000))})


@app.route("/model/explain", methods=["GET"])
def model_explain_info():
    return jsonify({
        "shap_available": SHAP_AVAILABLE,
        "explanation": "The /predict response includes shap_explanation with top contributing features. If SHAP is unavailable, a safe fallback explanation is used.",
        "features": FEATURE_COLS
    })


@app.route("/demo/high-risk", methods=["GET"])
def demo_high_risk():
    sample = {
        "location_name": "Demo Forest Zone",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "temperature_c": 38,
        "humidity_percent": 25,
        "wind_speed_kmh": 45,
        "air_quality_index": 180,
        "smoke_detected": 1,
        "sensor_risk_index": 0.88,
        "power_line_risk": 0.82,
        "sparking_probability": 0.80,
        "human_activity_index": 0.76,
        "vegetation_density": 0.90,
        "vegetation_dryness": 0.85,
        "campfire_reports": 3,
        "illegal_burning_reports": 1,
        "crowd_density": 90
    }

    result = predict_wildfire_from_data(sample)

    if result["auto_alert_required"]:
        alert = create_alert(
            alert_type="WILDFIRE_RISK",
            latitude=result["location"]["latitude"],
            longitude=result["location"]["longitude"],
            message=f"Demo auto alert: {result['risk_level']} wildfire risk detected at {result['location']['location_name']}.",
            risk_level=result["risk_level"],
            priority=result["decision"]["final_priority"],
            source="DEMO_HIGH_RISK",
            explanations=result["explanations"],
            recommended_action=result["recommended_action"],
            extra={
                "prediction_confidence": result["confidence"],
                "emergency_score": result["decision"]["emergency_score"]
            }
        )
        result["alert_created"] = True
        result["alert_id"] = alert["alert_id"]
    else:
        result["alert_created"] = False
        result["alert_id"] = None

    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  WILDFIRE SYSTEM — BACKEND RUNNING WITH SQLITE + EMAIL SIMULATION")
    print("  User Dashboard : http://localhost:5000")
    print("  Admin Dashboard: http://localhost:5000/admin")
    print("  Health Check   : http://localhost:5000/health")
    print("  Demo High Risk : http://localhost:5000/demo/high-risk")
    print(f"  Database       : {DB_PATH}")
    print("=" * 70 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
