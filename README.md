# Document Analysis Pipeline (MVP)

## Run local (Docker)
```bash
docker compose up --build
```
## Levantar el stack
```bash

```

## recontruir y levantar En Windows
```bash
docker compose down
docker compose build
docker compose up
```

## recontruir y levantar en Ubuntu
```bash
docker-compose down
docker-compose build
docker-compose up
```

## Verificar la API

* Health: http://localhost:8000/v1/healthz → {"status":"ok"}
* Docs (Swagger): http://localhost:8000/docs
* Métricas Prometheus: http://localhost:8000/metrics

## Indexar (POST /v1/index)
```bash
curl -X POST http://localhost:8000/v1/index
```

Entrenar :
```bash
curl -X POST http://localhost:8000/v1/classifier/train -H "Content-Type: application/json" -d "{}"
```

* El modelo se guarda en: data/model/classifier.joblib

Local (Sin Docker)
```bash
python -m venv .venv
. .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn app.main:app --reload
```

