from fastapi import FastAPI
from app.api.v1.routes_health import router as health_router
from app.api.v1.routes_search import router as search_router
from app.api.v1.routes_ner import router as ner_router
from app.api.v1.routes_classifier import router as clf_router

# ⬇️ NUEVO
from app.api.v1.routes_jobs import router as jobs_router
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Document Analysis Pipeline", version="0.5.0")

#Routers
app.include_router(health_router, prefix="/v1")
app.include_router(search_router, prefix="/v1")
app.include_router(ner_router, prefix="/v1")
app.include_router(clf_router, prefix="/v1")
app.include_router(jobs_router, prefix="/v1")

#MÉTRICAS /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
