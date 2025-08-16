from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.ml.ner_pipeline import NERService
from app.services.keyfacts import extract_key_facts
# Reutilizamos el índice y textos de la fase 2:
from app.api.v1.routes_search import _STATE

router = APIRouter()

# Cargamos el NER una sola vez
_NER = NERService()

class NEROut(BaseModel):
    text: str
    ents: List[Dict[str, Any]]
    facts: Dict[str, Any]

@router.get("/ner", response_model=NEROut)
def ner_text(text: str = Query(..., min_length=1)):
    ents = _NER.analyze(text)
    facts = extract_key_facts(text, ents)
    return {"text": text[:5000], "ents": ents, "facts": facts}

@router.get("/search_ner", response_model=List[NEROut])
def search_with_ner(q: str = Query(..., min_length=1), k: int = 5, facts: bool = True):
    """
    1) Busca semánticamente (ya indexado con /v1/index)
    2) Corre NER (y key facts) sobre los mejores resultados
    """
    if not _STATE["ready"]:
        raise HTTPException(status_code=400, detail="Primero crea el índice con POST /v1/index")

    emb = _STATE["embedder"]
    idx = _STATE["index"]
    texts = _STATE["texts"]

    q_vec = emb.embed([q])
    hits = idx.search(q_vec, k=k)[0]

    out: List[NEROut] = []
    for h in hits:
        snippet = texts[h["index"]]
        ents = _NER.analyze(snippet)
        facts_obj = extract_key_facts(snippet, ents) if facts else {}
        out.append(NEROut(text=snippet[:2000], ents=ents, facts=facts_obj))
    return out
