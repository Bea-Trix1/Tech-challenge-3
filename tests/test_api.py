import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health(use_dummy_onnx):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_valid_payload(use_dummy_onnx):
    resp = client.post("/predict", json={"text": "malignant tumor with metastasis"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "urgente"
    assert all(0 <= p <= 1 for p in body["probabilities"].values())


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": 123}, {"text": "x" * 20_001}])
def test_predict_invalid_payload(payload, use_dummy_onnx):
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_reports_latency_headers(use_dummy_onnx):
    resp = client.post("/predict", json={"text": "malignant tumor with metastasis"})
    assert resp.status_code == 200
    assert float(resp.headers["x-inference-time-ms"]) >= 0
    assert float(resp.headers["x-process-time-ms"]) >= 0


def test_latency_metrics_aggregate_predictions(use_dummy_onnx):
    client.post("/predict", json={"text": "malignant tumor with metastasis"})
    body = client.get("/metrics/latency").json()
    assert body["inference"]["count"] >= 1
    assert body["http"]["count"] >= 1
    assert body["inference"]["p95_ms"] >= body["inference"]["p50_ms"]
    assert body["inference"]["window"] > 0
