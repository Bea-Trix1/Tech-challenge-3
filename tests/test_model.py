import numpy as np
import pandas as pd
import pytest

from src.api import get_session, predict
from src.train import URGENCY_MAP, load_data, save_model


def test_load_data_maps_all_classes_to_urgency(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame({"condition_label": [1, 2, 3, 4, 5], "medical_abstract": list("abcde")}).to_csv(
        csv, index=False
    )
    df = load_data(csv)
    assert list(df.columns) == ["text", "label"]
    assert df["label"].tolist() == [URGENCY_MAP[i] for i in range(1, 6)]


def test_load_data_rejects_wrong_schema(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame({"foo": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Colunas ausentes"):
        load_data(csv)


def test_save_model_writes_file(dummy_model, tmp_path):
    assert save_model(dummy_model, tmp_path / "m.joblib").stat().st_size > 0


def test_onnx_matches_sklearn(dummy_model, dummy_df, use_dummy_onnx):
    labels, probs = get_session().run(None, {"text": dummy_df[["text"]].to_numpy()})
    assert (labels == dummy_model.predict(dummy_df["text"])).all()
    np.testing.assert_allclose(probs, dummy_model.predict_proba(dummy_df["text"]), atol=1e-3)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("malignant carcinoma with metastasis", "urgente"),
        ("recurrent seizure and epilepsy", "atencao"),
        ("mild fever on routine checkup", "normal"),
    ],
)
def test_predict_returns_expected_label(text, expected, use_dummy_onnx):
    out = predict(text)
    assert out.label == expected
    assert set(out.probabilities) == set(URGENCY_MAP.values())
    assert sum(out.probabilities.values()) == pytest.approx(1, abs=1e-3)
