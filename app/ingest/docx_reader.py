from typing import List
from app.ingest.types import DocChunk, IngestResult
import os, uuid
from docx import Document as Docx

def read_docx(path: str) -> IngestResult:
    doc_id = str(uuid.uuid4())
    doc = Docx(path)
    chunks: List[DocChunk] = []
    for i, para in enumerate(doc.paragraphs):
        text = para.text or ""
        if text.strip():
            chunks.append({"text": text, "meta": {"paragraph": i + 1}})
    return {
        "doc_id": doc_id,
        "chunks": chunks,
        "metadata": {"filename": os.path.basename(path), "format": "docx"},
    }
