"""DAG de retreino: ler CSV -> treinar -> salvar modelo (joblib + ONNX servido pela API).

Reaproveita as funções de src/train.py. Entre as tasks trafegam só caminhos (XCom), nunca
DataFrames ou modelos.
"""

from datetime import datetime, timedelta

from airflow.sdk import dag, task


@dag(
    dag_id="train_pipeline",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["mlops", "triagem"],
)
def train_pipeline():
    @task
    def ler_csv() -> str:
        from src.train import ROOT, TRAIN_CSV, download_data, load_data

        download_data()
        out = ROOT / "data" / "processed" / "train.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df = load_data(TRAIN_CSV)
        df.to_csv(out, index=False)  # sobrescreve: reexecução idempotente
        print(f"{len(df)} amostras: {df['label'].value_counts().to_dict()}")
        return str(out)

    @task
    def treinar(csv_path: str) -> str:
        import json

        import pandas as pd

        from src.train import METRICS_PATH, TEST_CSV, evaluate, load_data, save_model, train

        model = train(pd.read_csv(csv_path))
        report = evaluate(model, load_data(TEST_CSV))
        METRICS_PATH.write_text(json.dumps(report, indent=2))
        print(f"accuracy={report['accuracy']:.4f}")
        return str(save_model(model))

    @task
    def salvar_modelo(model_path: str) -> str:
        import joblib

        from src.export_onnx import export_onnx

        return str(export_onnx(joblib.load(model_path)))

    salvar_modelo(treinar(ler_csv()))


train_pipeline()
