from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from src import api


def count(metric, labels):
    return REGISTRY.get_sample_value(metric, labels) or 0


def test_metrics_records_predictions_and_validation_errors(use_dummy_onnx):
    client = TestClient(api.app)
    success = {"method": "POST", "route": "/predict", "status": "200"}
    invalid = {**success, "status": "422"}
    duration = {"method": "POST", "route": "/predict"}
    before_success = count("triagem_http_requests_total", success)
    before_invalid = count("triagem_http_requests_total", invalid)
    before_duration = count("triagem_http_request_duration_seconds_count", duration)

    assert client.post("/predict", json={"text": "malignant tumor"}).status_code == 200
    assert client.post("/predict", json={"text": ""}).status_code == 422

    assert count("triagem_http_requests_total", success) == before_success + 1
    assert count("triagem_http_requests_total", invalid) == before_invalid + 1
    assert count("triagem_http_request_duration_seconds_count", duration) == before_duration + 2
    assert count("triagem_http_request_duration_seconds_sum", duration) > 0
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "triagem_http_request_duration_seconds_bucket" in response.text
    assert "malignant tumor" not in response.text
    assert 'route="/metrics"' not in response.text


def test_unhandled_error_is_counted(monkeypatch):
    def fail(_text):
        raise RuntimeError("Falha simulada")

    monkeypatch.setattr(api, "predict", fail)
    labels = {"method": "POST", "route": "/predict", "status": "500"}
    before = count("triagem_http_requests_total", labels)
    with TestClient(api.app, raise_server_exceptions=False) as client:
        assert client.post("/predict", json={"text": "example"}).status_code == 500
    assert count("triagem_http_requests_total", labels) == before + 1


def test_unknown_paths_share_one_label():
    labels = {"method": "GET", "route": "unmatched", "status": "404"}
    before = count("triagem_http_requests_total", labels)
    client = TestClient(api.app)
    assert client.get("/not-found-one").status_code == 404
    assert client.get("/not-found-two").status_code == 404
    assert count("triagem_http_requests_total", labels) == before + 2
