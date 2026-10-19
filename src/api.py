"""API de inferência: recebe o texto do laudo e devolve a classe de urgência.

Uso: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import json
import os
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.export_onnx import ONNX_PATH


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class PredictResponse(BaseModel):
    label: str
    probabilities: dict[str, float]


@lru_cache
def get_session() -> ort.InferenceSession:
    path = os.getenv("MODEL_PATH", str(ONNX_PATH))
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def predict(text: str) -> PredictResponse:
    sess = get_session()
    classes = json.loads(sess.get_modelmeta().custom_metadata_map["classes"])
    label, probs = sess.run(None, {"text": np.array([[text]])})
    return PredictResponse(
        label=str(label[0]),
        probabilities={c: round(float(p), 4) for c, p in zip(classes, probs[0], strict=True)},
    )


app = FastAPI(title="Triagem de Laudos", version="0.1.0")


@app.get("/health")
def health() -> dict:
    get_session()  # falha cedo se o modelo não existir
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> PredictResponse:
    return predict(req.text)
