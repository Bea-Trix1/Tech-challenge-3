"""Benchmark de latência: pipeline sklearn original vs ONNX Runtime.

Mede inferência unitária (1 laudo por chamada, o caso da API) com percentis P50/P95/P99,
e throughput em lote. Uso: python -m src.benchmark
"""

import json
import platform
import time

import joblib
import numpy as np
import onnxruntime as ort

from src.export_onnx import ONNX_PATH
from src.train import MODEL_PATH, ROOT, TEST_CSV, load_data

N_SINGLE = 1000
WARMUP = 50
RESULTS_JSON = ROOT / "metrics" / "benchmark.json"
RESULTS_MD = ROOT / "metrics" / "benchmark.md"


def time_calls(fn, inputs) -> np.ndarray:
    for x in inputs[:WARMUP]:
        fn(x)
    times = []
    for x in inputs:
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def summarize(ms: np.ndarray, batch_s: float, n_batch: int) -> dict:
    return {
        "mean_ms": round(float(ms.mean()), 3),
        "p50_ms": round(float(np.percentile(ms, 50)), 3),
        "p95_ms": round(float(np.percentile(ms, 95)), 3),
        "p99_ms": round(float(np.percentile(ms, 99)), 3),
        "batch_throughput_per_s": round(n_batch / batch_s, 1),
    }


def main() -> None:
    texts = load_data(TEST_CSV)["text"].tolist()
    single = texts[:N_SINGLE]

    sk = joblib.load(MODEL_PATH)
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    def sk_one(t):
        return sk.predict_proba([t])

    def onnx_one(t):
        return sess.run(None, {"text": np.array([[t]])})

    batch = np.array(texts).reshape(-1, 1)
    t0 = time.perf_counter()
    sk.predict_proba(texts)
    sk_batch = time.perf_counter() - t0
    t0 = time.perf_counter()
    sess.run(None, {"text": batch})
    onnx_batch = time.perf_counter() - t0

    results = {
        "sklearn": summarize(time_calls(sk_one, single), sk_batch, len(texts)),
        "onnxruntime": summarize(time_calls(onnx_one, single), onnx_batch, len(texts)),
        "model_size_mb": {
            "sklearn": round(MODEL_PATH.stat().st_size / 1e6, 2),
            "onnxruntime": round(ONNX_PATH.stat().st_size / 1e6, 2),
        },
        "env": {
            "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
        },
        "n_single": N_SINGLE,
        "n_batch": len(texts),
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2))

    s, o = results["sklearn"], results["onnxruntime"]
    rows = [
        ("Latência média (ms)", "mean_ms"),
        ("P50 (ms)", "p50_ms"),
        ("P95 (ms)", "p95_ms"),
        ("P99 (ms)", "p99_ms"),
        ("Throughput em lote (laudos/s)", "batch_throughput_per_s"),
    ]
    lines = ["| Métrica | sklearn | ONNX Runtime | Ganho |", "|---|---:|---:|---:|"]
    for name, k in rows:
        gain = s[k] / o[k] if k != "batch_throughput_per_s" else o[k] / s[k]
        lines.append(f"| {name} | {s[k]} | {o[k]} | {gain:.2f}x |")
    sz = results["model_size_mb"]
    lines.append(f"| Tamanho do modelo (MB) | {sz['sklearn']} | {sz['onnxruntime']} | - |")
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
