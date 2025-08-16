from typing import List
from app.ingest.types import DocChunk, IngestResult
from pypdf import PdfReader
import os, uuid

def read_pdf(path: str) -> IngestResult:
    doc_id = str(uuid.uuid4())
    reader = PdfReader(path)
    chunks: List[DocChunk] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        chunks.append({"text": text, "meta": {"page": i + 1}})
    return {
        "doc_id": doc_id,
        "chunks": chunks,
        "metadata": {"filename": os.path.basename(path), "format": "pdf"},
    }
