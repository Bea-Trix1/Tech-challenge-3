"""Converte o pipeline sklearn (TF-IDF + LR) para ONNX.

Uso: python -m src.export_onnx
"""

import json
from pathlib import Path

import joblib
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline

from src.train import MODEL_PATH, ROOT

ONNX_PATH = ROOT / "models" / "model.onnx"


def export_onnx(model: Pipeline, path: Path = ONNX_PATH) -> Path:
    # O tokenizer ONNX precisa da mesma regex do TfidfVectorizer (default: \b\w\w+\b);
    # zipmap=False devolve as probabilidades como tensor, mais rápido que lista de dicts.
    onx = to_onnx(
        model,
        initial_types=[("text", StringTensorType([None, 1]))],
        options={
            id(model.named_steps["tfidf"]): {"tokenexp": r"\b\w\w+\b"},
            id(model): {"zipmap": False},
        },
        target_opset=17,
    )
    # Ordem das colunas de `probabilities`, lida pela API.
    meta = onx.metadata_props.add()
    meta.key, meta.value = "classes", json.dumps([str(c) for c in model.classes_])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onx.SerializeToString())
    return path


if __name__ == "__main__":
    print(export_onnx(joblib.load(MODEL_PATH)))
