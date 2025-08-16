from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.services.pipeline import build_index

router = APIRouter()

_STATE = {
    "ready": False,
    "index": None,
    "embedder": None,
    "texts": [],
    "meta": [],
    "data_dir": "data/samples",
}

class IndexIn(BaseModel):
    data_dir: Optional[str] = None

class SearchOutItem(BaseModel):
    score: float
    text: str
    meta: Dict[str, Any]

@router.post("/index")
def create_index(body: IndexIn = IndexIn()):
    data_dir = body.data_dir or _STATE["data_dir"]
    index, emb, texts, meta = build_index(data_dir=data_dir)
    _STATE.update({
        "ready": True,
        "index": index,
        "embedder": emb,
        "texts": texts,
        "meta": meta,
        "data_dir": data_dir,
    })
    return {"ok": True, "chunks_indexed": len(texts), "data_dir": data_dir}

@router.get("/search", response_model=List[List[SearchOutItem]])
def search(q: str = Query(..., min_length=1), k: int = 5):
    if not _STATE["ready"]:
        raise HTTPException(status_code=400, detail="Primero crea el índice con POST /v1/index")
    emb = _STATE["embedder"]
    idx = _STATE["index"]
    texts = _STATE["texts"]

    q_vec = emb.embed([q])
    hits = idx.search(q_vec, k=k)[0]

    out: List[SearchOutItem] = []
    for h in hits:
        out.append(SearchOutItem(
            score=h["score"],
            text=texts[h["index"]][:500],
            meta=h["meta"],
        ))
    return [out]
