# Contribuindo

## Commits: Conventional Commits

Formato: `<tipo>(<escopo opcional>): <descrição no imperativo, minúscula, sem ponto final>`

| Tipo | Quando usar |
|---|---|
| `feat` | nova funcionalidade (endpoint, task da DAG, técnica de otimização) |
| `fix` | correção de bug |
| `docs` | só documentação |
| `test` | adiciona ou ajusta testes |
| `ci` | workflows do GitHub Actions |
| `build` | Dockerfile, dependências (`pyproject.toml`/`poetry.lock`) |
| `refactor` | mudança de código sem alterar comportamento |
| `perf` | melhoria de performance/latência |
| `chore` | manutenção que não se encaixa acima |

Escopos usados no projeto: `model`, `api`, `airflow`, `monitoring`, `docker`.
Mudança incompatível: `!` após o tipo/escopo (`feat(api)!: ...`) e rodapé `BREAKING CHANGE: ...`.

Exemplos:

```
feat(model): exportar pipeline TF-IDF + LR para ONNX
fix(api): retornar 422 para texto vazio
ci: adicionar build da imagem Docker no workflow
```

## Branches

- `main` é protegida: nada de push direto.
- Trabalhe em `feature/<descricao-curta>` ou `fix/<descricao-curta>`, a partir da `main` atualizada.

## Pull Requests

1. Título do PR no padrão Conventional Commits (o CI valida, job `pr-title`).
2. Preencha o template (o quê, por quê, como testar).
3. Para fazer merge: CI verde (lint → test → build) + **1 aprovação** de outra pessoa do grupo.
4. PRs pequenos e focados: um assunto por PR.
5. Merge via **squash**: o título do PR vira o commit na `main`, mantendo o histórico semântico.
6. Mudou o modelo? Inclua no PR as métricas (`metrics/train_metrics.json`) e o benchmark (`metrics/benchmark.md`).

## Antes de abrir o PR

```bash
poetry run ruff check . && poetry run ruff format --check .
poetry run pytest
```
