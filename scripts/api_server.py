# scripts/api_server.py
import os
import io
import numpy as np
import joblib
import librosa

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# If your analytics router lives in scripts/analytics_routes.py
# (and you created scripts/__init__.py)
from scripts.analytics_routes import router as analytics_router


# -----------------------------
# Paths (robust: works from any cwd)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../UrbanSoundscape/scripts
PROJECT_DIR = os.path.dirname(BASE_DIR)                    # .../UrbanSoundscape

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "sound_rf_multilabel.pkl")
LABELS_PATH = os.path.join(PROJECT_DIR, "models", "label_names.txt")
TMP_WAV_PATH = os.path.join(BASE_DIR, "temp.wav")

CHART_TOP_LABELS = os.path.join(PROJECT_DIR, "results", "chart_top_labels.png")
CHART_PRED_COUNT = os.path.join(PROJECT_DIR, "results", "chart_predicted_count_distribution.png")
CHART_TOP_PAIRS = os.path.join(PROJECT_DIR, "results", "chart_top_pairs.png")


# -----------------------------
# Load model + labels at startup
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Label names file not found at: {LABELS_PATH}")

with open(LABELS_PATH, "r") as f:
    LABEL_NAMES = [line.strip() for line in f if line.strip()]

if not LABEL_NAMES:
    raise ValueError("LABEL_NAMES is empty. Check models/label_names.txt")


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Urban Sound AI")

# Optional: allow frontend later (dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach analytics routes under /analytics
app.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])


# -----------------------------
# Helpers
# -----------------------------
def extract_mfcc_from_bytes(wav_bytes: bytes, sr: int = 22050, duration: float = 5.0, n_mfcc: int = 40) -> np.ndarray:
    """
    Writes bytes to a temp wav then loads via librosa (simple + reliable).
    Returns shape (1, n_mfcc).
    """
    with open(TMP_WAV_PATH, "wb") as f:
        f.write(wav_bytes)

    y, _sr = librosa.load(TMP_WAV_PATH, sr=sr, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=_sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)  # (1, 40)
    return mfcc_mean


def get_probs_for_multilabel(model_obj, X: np.ndarray):
    """
    MultiOutputClassifier.predict_proba returns a list of arrays:
    probs[i] -> array of shape (n_samples, n_classes_for_label) usually 2.
    We take probability of class '1' if available, else fallback to last column.
    Returns list[float] length = n_labels
    """
    probs = model_obj.predict_proba(X)
    out = []
    for p in probs:
        # p shape typically (n_samples, 2). Want prob of class 1.
        if p.shape[1] >= 2:
            out.append(float(p[0, 1]))
        else:
            out.append(float(p[0, -1]))
    return out


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "Urban Sound AI", "labels": len(LABEL_NAMES)}


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        X = extract_mfcc_from_bytes(contents)

        probs_1 = get_probs_for_multilabel(model, X)

        results = {}
        for i, label_name in enumerate(LABEL_NAMES):
            prob = float(probs_1[i])
            pred = int(prob >= 0.5)
            results[label_name] = {
                "prediction": pred,
                "confidence": round(prob, 4),
            }

        return {"prediction": results}

    except Exception as e:
        # Return a readable error in Swagger instead of silent 500
        return JSONResponse(status_code=500, content={"error": str(e)})

# ===============================
# 🔽 PASTE NEW ENDPOINT BELOW 🔽
# ===============================

from pydantic import BaseModel
from typing import List

class MFCCRequest(BaseModel):
    mfcc: List[float]
    threshold: float = 0.2
    top_k: int = 5


@app.post("/predict_mfcc", tags=["IoT"])
async def predict_mfcc(req: MFCCRequest):

    if len(req.mfcc) != 40:
        return {"error": f"Expected 40 MFCC values, got {len(req.mfcc)}"}

    X = np.array(req.mfcc, dtype=float).reshape(1, -1)

    probs_1 = get_probs_for_multilabel(model, X)

    scored = []
    for i, label_name in enumerate(LABEL_NAMES):
        prob = float(probs_1[i])
        scored.append((label_name, prob))

    filtered = [(lbl, p) for lbl, p in scored if p >= req.threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)

    top = filtered[: req.top_k]

    return {
        "threshold": req.threshold,
        "top_k": req.top_k,
        "predicted": [
            {"label": lbl, "confidence": round(p, 4)} for lbl, p in top
        ],
        "predicted_count": len(filtered),
    }

# Optional: serve generated chart PNGs directly
@app.get("/charts/top-labels", tags=["Charts"])
def chart_top_labels():
    if not os.path.exists(CHART_TOP_LABELS):
        return JSONResponse(status_code=404, content={"error": f"Not found: {CHART_TOP_LABELS}"})
    return FileResponse(CHART_TOP_LABELS)


@app.get("/charts/predicted-count-distribution", tags=["Charts"])
def chart_predicted_count_distribution():
    if not os.path.exists(CHART_PRED_COUNT):
        return JSONResponse(status_code=404, content={"error": f"Not found: {CHART_PRED_COUNT}"})
    return FileResponse(CHART_PRED_COUNT)


@app.get("/charts/top-pairs", tags=["Charts"])
def chart_top_pairs():
    if not os.path.exists(CHART_TOP_PAIRS):
        return JSONResponse(status_code=404, content={"error": f"Not found: {CHART_TOP_PAIRS}"})
    return FileResponse(CHART_TOP_PAIRS)

