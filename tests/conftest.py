import pandas as pd
import pytest

from src import api
from src.export_onnx import export_onnx
from src.train import train

# Amostra sintética ("dummy training"): valida o pipeline sem depender do dataset real.
VOCAB = {
    "urgente": "malignant tumor carcinoma myocardial infarction metastasis",
    "atencao": "gastric ulcer bowel seizure epilepsy neuropathy",
    "normal": "general symptoms fever mild routine checkup",
}


@pytest.fixture(scope="session")
def dummy_df() -> pd.DataFrame:
    rows = [
        {"text": f"patient {i} presents {words}", "label": label}
        for label, words in VOCAB.items()
        for i in range(20)
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def dummy_model(dummy_df):
    return train(dummy_df)


@pytest.fixture(scope="session")
def onnx_path(dummy_model, tmp_path_factory):
    return export_onnx(dummy_model, tmp_path_factory.mktemp("models") / "model.onnx")


@pytest.fixture
def use_dummy_onnx(onnx_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(onnx_path))
    api.get_session.cache_clear()
    yield
    api.get_session.cache_clear()
