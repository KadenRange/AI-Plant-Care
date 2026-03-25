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

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from care.care_lookup import get_care
from model.adapters.efficientnet_adapter import EfficientNetAdapter
from model.adapters.vit_adapter import ViTAdapter
from model.gradcam import generate_heatmap
from model.registry import ModelRegistry

logger = logging.getLogger(__name__)

# ── Shared state (populated during lifespan startup) ─────────────────────────
registry:     ModelRegistry | None = None
classifier:   object | None        = None   # active adapter instance
model_loaded: bool                 = False


# ── Lifespan: build registry and load active adapter once at startup ──────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry, classifier, model_loaded

    registry = ModelRegistry()
    registry.register("efficientnet", EfficientNetAdapter)
    registry.register("vit",          ViTAdapter)

    try:
        registry.load_active()
        classifier   = registry.active()
        model_loaded = True
    except Exception:
        logger.exception("Failed to load active model adapter '%s'.", registry.active_name)

    yield

    classifier   = None
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


@app.get("/models")
def models():
    if registry is None:
        return {"active": None, "available": []}
    return {"active": registry.active_name, "available": registry.list()}


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


@app.post("/explain")
async def explain(
    file: UploadFile = File(...),
    class_idx: int | None = Query(None),
):
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

    overlay = generate_heatmap(image, classifier.model, class_idx=class_idx)

    buf = io.BytesIO()
    overlay.save(buf, format="JPEG")
    return Response(content=buf.getvalue(), media_type="image/jpeg")
