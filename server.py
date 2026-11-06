
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import Depends
import sqlite3
import hashlib
import secrets
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "fraud_model.pkl"
METRICS_PATH = BASE / "models" / "model_metrics.json"
WEB_PATH = BASE / "web"

app = FastAPI(title="FraudShield AI")
@app.get("/api/auth/me")
def current_user(request: Request):
    token = request.cookies.get("fraudshield_session")

    if not token:
        raise HTTPException(status_code=401, detail="Please login.")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        user = conn.execute("""
            SELECT users.id, users.name, users.email, users.role
            FROM users
            JOIN sessions ON users.id = sessions.user_id
            WHERE sessions.token = ?
        """, (token,)).fetchone()

    if not user:
        raise HTTPException(status_code=401, detail="Session expired.")

    return dict(user)
# ================= AUTHENTICATION =================
def require_login(request: Request):
    token = request.cookies.get("fraudshield_session")

    if not token:
        raise HTTPException(status_code=401, detail="Please login first.")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        user = conn.execute("""
            SELECT users.id, users.name, users.email, users.role
            FROM users
            JOIN sessions ON users.id = sessions.user_id
            WHERE sessions.token = ?
        """, (token,)).fetchone()

    if not user:
        raise HTTPException(status_code=401, detail="Session expired.")

    return dict(user)

DB_PATH = BASE / "users.db"

def init_auth_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT DEFAULT 'NEW USER'
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL
            )
        """)

init_auth_db()


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        200000
    ).hex()


@app.post("/api/auth/register")
def register(data: dict):
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not name or not email or len(password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Enter your name, valid email and password of at least 8 characters."
        )

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("""
                INSERT INTO users (name, email, password_hash, salt, role)
                VALUES (?, ?, ?, ?, 'NEW USER')
            """, (name, email, password_hash, salt))

            user_id = cursor.lastrowid

    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Email already registered.")

    token = secrets.token_urlsafe(32)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id) VALUES (?, ?)",
            (token, user_id)
        )

    response = JSONResponse({
        "message": "Registration successful",
        "user": {
            "name": name,
            "email": email,
            "role": "NEW USER"
        }
    })

    response.set_cookie(
        "fraudshield_session",
        token,
        httponly=True,
        samesite="lax"
    )

    return response


@app.post("/api/auth/login")
def login(data: dict):
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    requested_role = data.get("role", "NEW USER").strip().upper()

    if requested_role not in ["NEW USER", "DEVELOPER", "ADMIN"]:
        raise HTTPException(status_code=400, detail="Invalid account role.")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        user = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        ).fetchone()

    if not user or hash_password(password, user["salt"]) != user["password_hash"]:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if user["role"].upper() != requested_role:
        raise HTTPException(
            status_code=403,
            detail="This account is not authorised for the selected login."
        )

    token = secrets.token_urlsafe(32)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id) VALUES (?, ?)",
            (token, user["id"])
        )

    response = JSONResponse({
        "message": "Login successful",
        "user": {
            "name": user["name"],
            "email": user["email"],
            "role": user["role"]
        }
    })

    response.set_cookie(
        "fraudshield_session",
        token,
        httponly=True,
        samesite="lax"
    )

    return response
    response.set_cookie(
        "fraudshield_session",
        token,
        httponly=True,
        samesite="lax"
    )

    return response


@app.post("/api/auth/logout")
def logout(request: Request):
    token = request.cookies.get("fraudshield_session")

    if token:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "DELETE FROM sessions WHERE token = ?",
                (token,)
            )

    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie("fraudshield_session")
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

artifact = joblib.load(MODEL_PATH)

if isinstance(artifact, dict):
    model = artifact.get("model", artifact.get("best_model"))
    if model is None:
        raise ValueError("No model found inside fraud_model.pkl")

    FEATURES = artifact.get("features")
    THRESHOLD = float(artifact.get("threshold", 0.5))
    MODEL_NAME = artifact.get("model_name", "Random Forest")
else:
    model = artifact
    FEATURES = getattr(model, "feature_names_in_", None)
    THRESHOLD = 0.5
    MODEL_NAME = "Random Forest"

if FEATURES is None:
    FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

FEATURES = list(FEATURES)

metrics = {}
if METRICS_PATH.exists():
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        metrics = {}


class Transaction(BaseModel):
    features: dict[str, float]



def analyze_many(transactions):
    rows = [
        {feature: float(transaction.get(feature, 0))
         for feature in FEATURES}
        for transaction in transactions
    ]

    frame = pd.DataFrame(rows, columns=FEATURES)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[:, 1]

    elif hasattr(model, "decision_function"):
        scores = model.decision_function(frame)
        probabilities = 1 / (1 + np.exp(-scores))

    else:
        probabilities = model.predict(frame)

    results = []

    for probability in probabilities:
        probability = max(0.0, min(1.0, float(probability)))

        prediction = (
            "Fraudulent"
            if probability >= THRESHOLD
            else "Genuine"
        )

        if probability >= 0.80:
            risk = "CRITICAL"
        elif probability >= 0.50:
            risk = "HIGH"
        elif probability >= 0.20:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        results.append({
            "fraud_probability": round(probability * 100, 2),
            "prediction": prediction,
            "risk_level": risk
        })

    return results


def analyze(values):
    return analyze_many([values])[0]


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "model": MODEL_NAME,
        "feature_count": len(FEATURES)
    }


@app.get("/api/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "features": FEATURES,
        "threshold": THRESHOLD
    }


@app.get("/api/metrics")
def get_metrics():
    return metrics




@app.get("/api/model-info")
def model_info(user: dict = Depends(require_login)):
    return {
        "model_name": MODEL_NAME,
        "features": FEATURES,
        "threshold": THRESHOLD
    }


@app.get("/api/metrics")
def get_metrics(user: dict = Depends(require_login)):
    return metrics


@app.post("/api/predict")
def predict(data: Transaction, user: dict = Depends(require_login)):
    try:
        return analyze(data.features)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/api/batch")
def batch_predict(
    transactions: list[dict[str, float]],
    user: dict = Depends(require_login)
):
    if len(transactions) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10,000 rows per upload"
        )

    try:
        results = analyze_many(transactions)

        for index, result in enumerate(results):
            result["transaction_number"] = index + 1

        fraud_count = sum(
            item["prediction"] == "Fraudulent"
            for item in results
        )

        return {
            "total": len(results),
            "fraud_count": fraud_count,
            "genuine_count": len(results) - fraud_count,
            "results": results
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


if WEB_PATH.exists():
    app.mount("/", StaticFiles(directory=WEB_PATH, html=True), name="web")
