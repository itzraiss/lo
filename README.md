---
license: mit
title: Lotomania
sdk: docker
emoji: 🚀
---
# Lotomania – Sistema de Previsão, Geração de Jogos e Backtest (1999–2025)

Este projeto implementa um sistema automatizado, robusto e auditável para ingestão de dados da Lotomania, treinamento de modelos (ensemble), geração de até 3 jogos (50 números cada), simulação Monte-Carlo e backtest cronológico, com API (FastAPI) e dashboard (Streamlit).

## Aviso Importante
- Loterias são predominantemente aleatórias. O sistema NÃO promete ganhos. Ele estima probabilidades via simulação e só emite apostas automáticas quando critérios configuráveis são atendidos.
- Use resultados como apoio estatístico; não como garantia de retorno.

## Estrutura do Projeto

```
/workspace
  /src
    api.py
    backtest.py
    config.py
    db.py
    features.py
    ingest.py
    optimizer.py
    predict.py
    simulate.py
    train.py
    ui_streamlit.py
    update_from_api.py
    utils.py
  /data
  /logs
  /models
  results.xlsx                # histórico completo (se disponível)
  requirements.txt
  Dockerfile
  docker-compose.yml
  .env.example
  README.md
```

## Banco de Dados
- MongoDB
- DB: `lotomania`
- Collections: `contests`, `predictions`, `model_runs`, `logs`, `metrics`

### Esquemas
- `contests`
```json
{
  "contest": 2805,
  "date": "2025-08-10",
  "numbers": [x1, x2, ..., x20],
  "raw": {"...": "..."},
  "created_at": "ISODate(...)"
}
```
- `predictions`
```json
{
  "generated_at": "ISODate(...)",
  "model_version": "v2025-08-11-1",
  "p_i": {"1":0.02, "2":0.015, "...": 0.01},
  "candidate_pool": [1,2,...],
  "tickets": [[50 nums], [50 nums], [50 nums]],
  "simulations": {"n": 100000, "p_at_least_18": 0.003, "p_at_least_17": 0.02, "p_at_least_16": 0.12},
  "confidence_passed": false,
  "criteria": {"k": [16,17,18], "thresholds": {"18": 0.02, "17": 0.10, "16": 0.50}}
}
```

## Requisitos
- Python 3.10+
- Definir `MONGO_URI` no `.env`
- Colocar `results.xlsx` na raiz (se tiver histórico completo)

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env  # e configure MONGO_URI e thresholds
```

## Execução local

- Ingestão inicial do XLSX para Mongo:
```bash
python -m src.ingest --xlsx /workspace/results.xlsx
```

- Atualização via API (últimos resultados):
```bash
python -m src.update_from_api
```

- Treino + previsão + geração de tickets + simulação:
```bash
python -m src.train --retrain
python -m src.predict --generate --simulate --save
```

- API FastAPI:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

- Dashboard Streamlit:
```bash
streamlit run src/ui_streamlit.py
```

- Backtest cronológico (roll-forward):
```bash
python -m src.backtest --start-window 300 --step 1 --sims 50000
```

## Docker

- Build:
```bash
docker compose build
```
- Subir MongoDB + API + UI:
```bash
docker compose up -d
```

## Endpoints (API)
- `GET /health`
- `POST /predict` → Gera p_i, 3 tickets (ILP + fallback) e simulação; persiste em `predictions`
- `GET /latest` → Retorna última previsão persistida

## Critério de Emissão (configurável)
- `emitir` se qualquer condição verdadeira:
  - p̂(≥18) ≥ 0.02 OU p̂(≥17) ≥ 0.10 OU p̂(≥16) ≥ 0.50

## Explainability
- SHAP opcional (se instalado). Import opcional com fallback.

## Notas de Produção
- Scheduler diário: atualizar API, retrain, gerar previsões e logs
- Versão de modelo salva em `/models/{version}` com metadados
- Logs em `logs/`
- Monitorar concept drift (mudanças abruptas na distribuição de p_i)