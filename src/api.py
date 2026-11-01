"""API de inferência: recebe o texto do laudo e devolve a classe de urgência.

Uso: uvicorn src.api:app --host 0.0.0.0 --port 8000

Além do /predict, a API mede a própria latência (objetivo do projeto):
- header `X-Inference-Time-Ms` na resposta do /predict (só a inferência ONNX);
- header `X-Process-Time-Ms` em qualquer resposta (requisição HTTP completa);
- `GET /metrics/latency` com P50/P95/P99 das últimas N requisições.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field

logger = logging.getLogger("uvicorn.error")

# Não importamos src.export_onnx de propósito: ele puxa sklearn/skl2onnx/pandas.
# O serving só precisa do caminho do modelo, o que mantém a imagem enxuta e o boot rápido.
ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = ROOT / "models" / "model.onnx"

# Texto do warm-up: a 1ª inferência do ONNX Runtime é mais lenta (aloca arena/threads).
WARMUP_TEXT = "patient presents mild symptoms, routine checkup"
WARMUP_RUNS = 3

LATENCY_WINDOW = int(os.getenv("LATENCY_WINDOW", "1000"))


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class PredictResponse(BaseModel):
    label: str
    probabilities: dict[str, float]


class LatencySummary(BaseModel):
    count: int
    window: int
    mean_ms: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    max_ms: float | None = None


class LatencyStats:
    """Janela deslizante de latências, thread-safe (endpoints síncronos usam threadpool)."""

    def __init__(self, window: int) -> None:
        self.window = window
        self._samples: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, ms: float) -> None:
        with self._lock:
            self._samples.append(ms)

    def summary(self) -> LatencySummary:
        with self._lock:
            samples = np.fromiter(self._samples, dtype=np.float64)
        if samples.size == 0:
            return LatencySummary(count=0, window=self.window)
        return LatencySummary(
            count=int(samples.size),
            window=self.window,
            mean_ms=round(float(samples.mean()), 3),
            p50_ms=round(float(np.percentile(samples, 50)), 3),
            p95_ms=round(float(np.percentile(samples, 95)), 3),
            p99_ms=round(float(np.percentile(samples, 99)), 3),
            max_ms=round(float(samples.max()), 3),
        )


INFERENCE_LATENCY = LatencyStats(LATENCY_WINDOW)
HTTP_LATENCY = LatencyStats(LATENCY_WINDOW)


def _session_options() -> ort.SessionOptions:
    """Modelo é TF-IDF + LR (microssegundos): paralelismo de threads só adiciona overhead."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = int(os.getenv("ORT_INTRA_OP_THREADS", "1"))
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.log_severity_level = 3  # warnings/errors apenas
    return opts


@lru_cache
def get_session() -> ort.InferenceSession:
    path = Path(os.getenv("MODEL_PATH", str(ONNX_PATH)))
    if not path.is_file():
        raise FileNotFoundError(
            f"Modelo ONNX não encontrado em '{path}'. Rode `python -m src.export_onnx` "
            "ou defina MODEL_PATH."
        )
    return ort.InferenceSession(
        str(path), sess_options=_session_options(), providers=["CPUExecutionProvider"]
    )


@lru_cache
def get_classes() -> tuple[str, ...]:
    """Ordem das colunas de `probabilities`; lida uma vez, não a cada requisição."""
    meta = get_session().get_modelmeta().custom_metadata_map
    return tuple(json.loads(meta["classes"]))


def reset_model_cache() -> None:
    """Limpa sessão + classes (testes e troca de modelo em runtime)."""
    get_session.cache_clear()
    get_classes.cache_clear()


def predict(text: str) -> PredictResponse:
    sess = get_session()
    classes = get_classes()
    label, probs = sess.run(None, {"text": np.array([[text]])})
    return PredictResponse(
        label=str(label[0]),
        probabilities={c: round(float(p), 4) for c, p in zip(classes, probs[0], strict=True)},
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Falha cedo no boot do container se o modelo estiver ausente ou corrompido.
    get_classes()
    for _ in range(WARMUP_RUNS):
        predict(WARMUP_TEXT)
    logger.info(
        "Modelo ONNX carregado e aquecido (modelo=%s, classes=%s)",
        os.getenv("MODEL_PATH", str(ONNX_PATH)),
        list(get_classes()),
    )
    yield


app = FastAPI(title="Triagem de Laudos", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def measure_request_latency(request: Request, call_next):
    """Latência HTTP completa, medida em volta de todo o stack do FastAPI."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"
    if request.url.path == "/predict" and response.status_code == 200:
        HTTP_LATENCY.record(elapsed_ms)
    return response


@app.get("/health")
def health() -> dict:
    get_classes()  # falha cedo se o modelo não existir
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest, response: Response) -> PredictResponse:
    t0 = time.perf_counter()
    result = predict(req.text)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    INFERENCE_LATENCY.record(elapsed_ms)
    response.headers["X-Inference-Time-Ms"] = f"{elapsed_ms:.3f}"
    return result


@app.get("/metrics/latency")
def latency_metrics() -> dict[str, LatencySummary]:
    """Percentis das últimas LATENCY_WINDOW requisições (inferência pura e HTTP completo)."""
    return {"inference": INFERENCE_LATENCY.summary(), "http": HTTP_LATENCY.summary()}
