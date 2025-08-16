from typing import TypedDict, Dict, Any, List

class DocChunk(TypedDict):
    text: str
    meta: Dict[str, Any]

class IngestResult(TypedDict):
    doc_id: str
    chunks: List[DocChunk]
    metadata: Dict[str, Any]
