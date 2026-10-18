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
