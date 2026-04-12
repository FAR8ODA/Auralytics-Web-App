"""
backend/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI backend for the Auralytics demo.

Endpoints:
  GET  /health          liveness check
  GET  /machines        list available machine types + metadata
  POST /predict         run inference on uploaded audio

CORS is open so the Netlify frontend can call freely.

Usage:
  cd backend
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import ModelRegistry, predict, MODEL_CONFIGS, MachineType

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Auralytics API",
    description="Acoustic anomaly detection for industrial machines.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Netlify URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model registry — loaded once at startup ───────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"
registry   = ModelRegistry(MODELS_DIR)


@app.on_event("startup")
async def startup():
    print("\nAuralytics backend starting...")
    registry.load_all()
    print("All models loaded. Ready.\n")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(MODEL_CONFIGS.keys())}


@app.get("/machines")
def machines():
    """Return metadata for all available machine types."""
    return {
        machine: {
            "label":     cfg["label"],
            "auc":       cfg["auc"],
            "threshold": cfg["threshold"],
        }
        for machine, cfg in MODEL_CONFIGS.items()
    }


@app.post("/predict")
async def predict_endpoint(
    file:    UploadFile = File(...,  description="Audio file (.wav)"),
    machine: str        = Form(...,  description="Machine type: fan | pump | valve"),
):
    # Validate machine type
    if machine not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown machine '{machine}'. Choose from: {list(MODEL_CONFIGS.keys())}"
        )

    # Validate file type
    if file.content_type not in ("audio/wav", "audio/wave", "audio/x-wav",
                                  "application/octet-stream"):
        # Be lenient — some browsers send wrong MIME for .wav
        if not (file.filename or "").lower().endswith(".wav"):
            raise HTTPException(
                status_code=400,
                detail="Only .wav files are supported."
            )

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    try:
        result = predict(audio_bytes, machine, registry)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return result
