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
docker-compose.yml           # API + Prometheus + Grafana
monitoring/                  # coleta, datasource e dashboard provisionados
scripts/generate_traffic.py  # chamadas válidas e erros 422 para demonstração
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

### Latência da API

A API mede a própria latência (em `src/api.py`):

- `X-Inference-Time-Ms`: só a inferência ONNX, devolvido em cada `/predict`;
- `X-Process-Time-Ms`: requisição HTTP completa, devolvido em qualquer rota;
- `GET /metrics/latency`: P50/P95/P99 das últimas `LATENCY_WINDOW` (default 1000) requisições.

```bash
docker compose -f docker/fastapi/docker-compose.yml up --build
curl -s localhost:8000/metrics/latency | jq
```

O container de serving usa 1 worker, `ORT_INTRA_OP_THREADS=1`, sem `--reload`, monta
`models/` como somente leitura (o modelo retreinado pelo Airflow é servido sem rebuild) e
usa a mesma imagem do `docker/Dockerfile` (a que o CI testa).

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

## Monitoramento local: Prometheus + Grafana

Na raiz do repositório, com o Docker em execução:

```bash
docker compose up --build -d --wait
docker compose ps
python3 scripts/generate_traffic.py
```

Se já houver uma API usando a porta 8000, pare esse container/processo antes de subir
a stack. O modelo ONNX versionado é usado pela API; não é necessário treinar novamente.

| Serviço | Endereço | Acesso |
|---|---|---|
| API / Swagger | http://localhost:8000/docs | Sem autenticação local |
| Métricas | http://localhost:8000/metrics | Formato Prometheus |
| Prometheus | http://localhost:9090/targets | Target `triagem-api` deve estar `UP` |
| Grafana | http://localhost:3000/d/triagem-api | Usuário `admin`, senha inicial `admin` |

Para escolher outra senha inicial, execute `GRAFANA_ADMIN_PASSWORD=sua-senha docker compose up --build -d --wait`
na primeira inicialização. O Grafana persiste usuários e senha no volume; alterar essa
variável após criar o volume não redefine uma senha existente. Os três serviços são
publicados somente em localhost.

O datasource Prometheus e o dashboard **Triagem de Laudos — API**, na pasta
**Tech Challenge**, são provisionados automaticamente a partir de `monitoring/grafana/`.
O dashboard contém três painéis, todos filtrados para `/predict`:

1. Total de requisições por status HTTP, desde o início do processo da API.
2. Latência HTTP P50/P95, em segundos, estimada pelo histograma.
3. Taxa de erro: proporção de respostas 4xx e 5xx sobre o total de chamadas.

A API expõe `triagem_http_requests_total` (método, rota e status) e
`triagem_http_request_duration_seconds` (método e rota). A medição inclui validação,
inferência e envio da resposta. `/metrics` é excluído para que a coleta não gere seu
próprio tráfego; rotas inexistentes compartilham o rótulo `unmatched`. Textos dos laudos
não são colocados nos rótulos. Mantenha um processo Uvicorn por container nesta
configuração, pois os contadores ficam na memória do processo.

### Demonstração e evidência da entrega

O script usa apenas a biblioteca padrão do Python. Por padrão, faz 300 chamadas em
cerca de 150 segundos: 270 predições válidas (HTTP 200) e 30 entradas vazias, que devem
retornar HTTP 422. Esses erros de validação são intencionais para preencher o painel
de taxa de erro. Uma resposta inesperada faz o script terminar com erro.

```bash
python3 scripts/generate_traffic.py --requests 300 --interval 0.5
curl http://localhost:8000/metrics
```

A coleta ocorre a cada cinco segundos. Aguarde ao menos duas coletas após iniciar o
tráfego para visualizar taxas e percentis; o Grafana atualiza a cada cinco segundos.
Durante tráfego contínuo, a taxa de erro deve ficar próxima de 10%. Após o fim das
chamadas, a taxa cai para zero quando a janela de cálculo deixa de conter tráfego.
O total pode incluir chamadas de execuções anteriores; os contadores zeram ao reiniciar
a API. O healthcheck usa `/health`, que não entra nos painéis filtrados para `/predict`.

Para a entrega, o JSON está em
[`monitoring/grafana/dashboards/triagem.json`](monitoring/grafana/dashboards/triagem.json).
Abra o dashboard durante o tráfego para capturar um print ou demonstrá-lo no vídeo.
As configurações seguem a documentação de
[histogramas do cliente Prometheus](https://prometheus.github.io/client_python/instrumenting/histogram/)
e [provisionamento do Grafana](https://grafana.com/docs/grafana/latest/administration/provisioning/).

Para inspecionar problemas e encerrar os serviços:

```bash
docker compose logs --tail=100 api prometheus grafana
docker compose down
```

O encerramento preserva os volumes de métricas e do Grafana. O Compose do Airflow
continua separado em `docker/airflow/docker-compose.yml`.
