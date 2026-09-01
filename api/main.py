"""FastAPI inference service for img2GPS.

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from img2gps.inference import load_inference_model, predict_image  # noqa: E402


app = FastAPI(
    title="img2GPS API",
    description="Image-based GPS prediction using grid localization and soft top-k decoding.",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_model_cached():
    checkpoint_path = Path(os.getenv("IMG2GPS_CHECKPOINT", ROOT / "checkpoints" / "best.pt"))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return load_inference_model(checkpoint_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5):
    try:
        model = get_model_cached()
        result = predict_image(model, file.file, top_k=top_k)
        return result.to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
