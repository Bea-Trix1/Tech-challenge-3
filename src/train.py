"""Treino do classificador de urgência (TF-IDF + Logistic Regression).

Uso: python -m src.train
"""

import json
import urllib.request
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_URL = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main"
TRAIN_CSV = ROOT / "data" / "raw" / "medical_tc_train.csv"
TEST_CSV = ROOT / "data" / "raw" / "medical_tc_test.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "metrics" / "train_metrics.json"
SEED = 42

# O corpus traz 5 categorias de doença, não urgência. Heurística de triagem definida pelo
# grupo (não é rótulo clínico): 1 neoplasias, 4 cardiovascular -> urgente;
# 2 digestivo, 3 nervoso -> atencao; 5 condições patológicas gerais -> normal.
URGENCY_MAP = {1: "urgente", 4: "urgente", 2: "atencao", 3: "atencao", 5: "normal"}


def download_data() -> None:
    """Baixa os CSVs do repositório oficial do corpus (espelhado no Kaggle) se não existirem."""
    for path in (TRAIN_CSV, TEST_CSV):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{DATA_URL}/{path.name}", path)


def load_data(csv_path: Path) -> pd.DataFrame:
    """Lê o CSV do corpus e devolve colunas `text` e `label` (urgência)."""
    df = pd.read_csv(csv_path)
    missing = {"condition_label", "medical_abstract"} - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")
    df = df.dropna(subset=["medical_abstract", "condition_label"])
    return pd.DataFrame(
        {"text": df["medical_abstract"], "label": df["condition_label"].map(URGENCY_MAP)}
    ).dropna()


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=50_000),
            ),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        ]
    )


def train(df: pd.DataFrame) -> Pipeline:
    return build_pipeline().fit(df["text"], df["label"])


def evaluate(model: Pipeline, df: pd.DataFrame) -> dict:
    return classification_report(df["label"], model.predict(df["text"]), output_dict=True)


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def main() -> None:
    download_data()
    model = train(load_data(TRAIN_CSV))
    report = evaluate(model, load_data(TEST_CSV))
    save_model(model)
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(f"accuracy={report['accuracy']:.4f} macro_f1={report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
