"""FastAPI backend for the Auralytics demo.

Endpoints:
- GET /health: liveness check
- GET /machines: available machine types and thresholds
- POST /predict: upload a wav file and receive anomaly results
"""

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import MODEL_CONFIGS, ModelRegistry, predict

app = FastAPI(
    title="Auralytics API",
    description="Acoustic anomaly detection for industrial machines.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path(__file__).parent / "models"
registry = ModelRegistry(MODELS_DIR)


@app.on_event("startup")
async def startup() -> None:
    print("\nAuralytics backend starting...")
    print("Model registry ready. Models load lazily on first prediction.\n")


@app.get("/")
def root() -> dict:
    return {
        "service": "Auralytics API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "machines": "/machines",
    }

@app.head("/")
def root_head() -> Response:
    return Response(status_code=200)


@app.head("/health")
def health_head() -> Response:
    return Response(status_code=200)

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "available_models": list(MODEL_CONFIGS.keys()),
        "loaded_models": registry.loaded_models(),
    }


@app.get("/machines")
def machines() -> dict:
    return {
        machine: {
            "label": cfg["label"],
            "auc": cfg["auc"],
            "threshold": cfg["threshold"],
        }
        for machine, cfg in MODEL_CONFIGS.items()
    }


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(..., description="Audio file (.wav)"),
    machine: str = Form(..., description="Machine type: fan | pump | valve"),
) -> dict:
    if machine not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown machine '{machine}'. Choose from: {list(MODEL_CONFIGS.keys())}",
        )

    allowed_types = {"audio/wav", "audio/wave", "audio/x-wav", "application/octet-stream"}
    if file.content_type not in allowed_types and not (file.filename or "").lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file received.")

    try:
        return predict(audio_bytes, machine, registry)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
