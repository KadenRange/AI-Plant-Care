"""api.py — FastAPI backend for the AI Plant Care classifier.

Endpoints:
    GET  /health   — liveness check, reports whether the model is loaded
    POST /predict  — accepts a multipart image, returns species + care JSON

Run::

    uvicorn api:app --reload
"""
from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from care.care_lookup import get_care
from model.classifier import PlantClassifier

logger = logging.getLogger(__name__)

# ── Class names — must match the order produced by the training dataloader ────
# The dataloader sorts labels alphabetically when using the __key__ path, so
# this list is sorted A-Z over the 7 canonical species names.
CLASS_NAMES: list[str] = [
    "Aloe Vera",
    "Dumb Cane (Dieffenbachia)",
    "Monstera Deliciosa",
    "Peace Lily (Spathiphyllum wallisii)",
    "Pothos (Epipremnum aureum)",
    "Snake Plant (Sansevieria)",
    "ZZ Plant (Zamioculcas zamiifolia)",
]

WEIGHTS_PATH = Path("model/weights/best_model.pth")

# ── Shared state (populated during lifespan startup) ─────────────────────────
classifier: PlantClassifier | None = None
model_loaded: bool = False


# ── Lifespan: load model once at startup ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, model_loaded

    if not WEIGHTS_PATH.exists():
        logger.warning(
            "Model weights not found at '%s'. "
            "/predict will return 503 until weights are present and the "
            "server is restarted.",
            WEIGHTS_PATH,
        )
    else:
        try:
            classifier = PlantClassifier(CLASS_NAMES, weights_path=WEIGHTS_PATH)
            model_loaded = True
            logger.info("PlantClassifier loaded from '%s'.", WEIGHTS_PATH)
        except Exception:
            logger.exception("Failed to load PlantClassifier.")

    yield

    # Cleanup (nothing to release for a CPU/GPU torch model)
    classifier = None
    model_loaded = False


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Plant Care API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_loaded or classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure model/weights/best_model.pth exists and restart the server.",
        )

    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    prediction = classifier.predict(image)
    care = get_care(prediction["species"])

    return {
        "species":    prediction["species"],
        "confidence": prediction["confidence"],
        "top3":       prediction["top3"],
        "care":       care,
    }
