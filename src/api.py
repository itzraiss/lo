from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from .db import get_latest_prediction, insert_prediction
from .predict import run_prediction

app = FastAPI(title="Lotomania Predictor", version="0.1.0")


class PredictRequest(BaseModel):
    generate: bool = True
    simulate: bool = True
    save: bool = True


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/predict")
async def predict(req: PredictRequest) -> Dict[str, Any]:
    res = run_prediction(req.generate, req.simulate, req.save)
    return res


@app.get("/latest")
async def latest() -> Dict[str, Any]:
    doc = get_latest_prediction()
    return {"latest": doc}