from typing import Iterable, List
from app.ingest.types import DocChunk

def simple_normalize(chunks: Iterable[DocChunk]) -> List[DocChunk]:
    out: List[DocChunk] = []
    for ch in chunks:
        text = " ".join((ch["text"] or "").split())
        out.append({"text": text, "meta": ch.get("meta", {})})
    return out
