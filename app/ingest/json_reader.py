from typing import List, Dict, Any
from app.ingest.types import DocChunk, IngestResult
import os, uuid, json

def read_json(path: str) -> IngestResult:
    doc_id = str(uuid.uuid4())
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    chunks: List[DocChunk] = [{"text": text, "meta": {"section": "root"}}]
    return {
        "doc_id": doc_id,
        "chunks": chunks,
        "metadata": {"filename": os.path.basename(path), "format": "json"},
    }
