# Tech-challenge-3

Triagem automática de laudos médicos por urgência (`normal` / `atencao` / `urgente`): classificador
TF-IDF + Logistic Regression exportado para ONNX e servido via FastAPI em Docker.

## Estrutura

```
src/train.py         # dados + treino (reaproveitado pela DAG e pelos testes)
src/export_onnx.py   # conversão sklearn -> ONNX (skl2onnx)
src/benchmark.py     # latência sklearn vs ONNX Runtime -> metrics/benchmark.md
src/api.py           # FastAPI: GET /health, POST /predict
dags/train_pipeline.py        # DAG Airflow: ler_csv -> treinar -> salvar_modelo
docker/Dockerfile             # imagem da API
docker/airflow/docker-compose.yml  # Airflow local para rodar a DAG
.github/workflows/ci.yml      # CI: lint -> test -> build (+ título de PR semântico)
```

## Dataset

[Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus) (o mesmo
do Kaggle; licença CC BY-SA 3.0): 14.438 resumos médicos (11.550 de treino e 2.888 de teste) em 5
categorias de doença. O corpus **não tem rótulo de urgência**, então o grupo definiu um mapeamento
heurístico, que **não é um rótulo clínico validado**:

| Categoria original | Urgência |
|---|---|
| 1 Neoplasms, 4 Cardiovascular diseases | `urgente` |
| 2 Digestive system diseases, 3 Nervous system diseases | `atencao` |
| 5 General pathological conditions | `normal` |

`python -m src.train` baixa os CSVs para `data/raw/` (fora do git) se eles ainda não existirem.

## Modelo

TF-IDF (unigramas + bigramas, `min_df=2`, 50k features) + Logistic Regression (`class_weight="balanced"`).
No conjunto de teste oficial: **acurácia 0.574 e macro-F1 0.557** (`metrics/train_metrics.json`).
Comparativo feito no mesmo split: Random Forest 0.48, LinearSVC 0.51. O teto baixo vem do próprio
corpus: a classe "general pathological conditions" se sobrepõe às demais, e mesmo o problema original
de 5 classes fica em ~0.52 com TF-IDF + LR.

## Otimização: ONNX Runtime

O pipeline inteiro (tokenização + TF-IDF + LR) é convertido com `skl2onnx`, e a API usa só o
`onnxruntime` em produção. Paridade conferida no conjunto de teste: 100% dos rótulos iguais aos do
sklearn e diferença máxima de probabilidade de 0.005 (o ONNX usa float32).

Benchmark (`python -m src.benchmark`, 1.000 laudos, 1 por chamada, CPU local, Python 3.11):

| Métrica | sklearn | ONNX Runtime | Ganho |
|---|---:|---:|---:|
| Latência média (ms) | 0.84 | 0.278 | 3.02x |
| P50 (ms) | 0.83 | 0.273 | 3.04x |
| P95 (ms) | 1.042 | 0.349 | 2.99x |
| P99 (ms) | 1.177 | 0.396 | 2.97x |
| Throughput em lote (laudos/s) | 5553.0 | 3736.8 | 0.67x |
| Tamanho do modelo (MB) | 3.26 | 2.08 | - |

O ONNX corta ~3x a latência por requisição, que é o cenário da API (real-time, 1 laudo por chamada).
Em lote grande (2.888 laudos de uma vez), o sklearn vetorizado ainda é mais rápido. Para um
eventual job batch, vale manter o sklearn. Quantização dinâmica não se aplica: a LR vira o operador
`LinearClassifier`, e não `MatMul`/`Gemm`, que são os operadores quantizáveis.

Baseline HTTP da API no Docker (300 requisições sequenciais, localhost): **p50 3.7 ms, p95 4.3 ms**.

## Como rodar

```bash
poetry install
poetry run python -m src.train        # treina -> models/model.joblib + metrics/train_metrics.json
poetry run python -m src.export_onnx  # -> models/model.onnx (servido pela API)
poetry run python -m src.benchmark    # -> metrics/benchmark.md
poetry run pytest

docker build -f docker/Dockerfile -t triagem-api .
docker run -p 8000:8000 triagem-api
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "Metastatic breast carcinoma treated with chemotherapy"}'
```

### Airflow (retreino)

```bash
docker compose -f docker/airflow/docker-compose.yml up
```

A UI fica em http://localhost:8080 (usuário `admin`, senha impressa no log). Dispare a DAG
`train_pipeline` (agendada `@weekly`). Ela grava `models/model.joblib`, `models/model.onnx` e as
métricas no próprio repositório montado. Para testar sem a UI:
`docker compose -f docker/airflow/docker-compose.yml run --rm airflow bash -c "airflow db migrate && airflow dags test train_pipeline"`.

## CI/CD

`.github/workflows/ci.yml` roda em push (`main`, `feature/**`, `fix/**`) e em PR para a `main`:

1. **lint**: `ruff check` + `ruff format --check`
2. **test**: `pytest` (API + função de predição + paridade sklearn/ONNX, com treino de sanidade em dados sintéticos, sem depender do dataset)
3. **build**: `docker build` + smoke test do container (`/health` e `/predict`)
4. **pr-title**: valida o título do PR no padrão Conventional Commits

O padrão de commits e as regras de PR estão em [CONTRIBUTING.md](CONTRIBUTING.md).
