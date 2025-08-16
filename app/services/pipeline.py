import os
from typing import List, Dict, Any, Tuple
from app.ingest.pdf_reader import read_pdf
from app.ingest.docx_reader import read_docx
from app.ingest.json_reader import read_json
from app.ingest.normalizer import simple_normalize
from app.ml.embeddings import EmbeddingService
from app.ml.index_faiss import SemanticIndex

def _read_any(path: str):
    lower = path.lower()
    if lower.endswith(".pdf"):
        return read_pdf(path)
    if lower.endswith(".docx"):
        return read_docx(path)
    if lower.endswith(".json"):
        return read_json(path)
    return None

def build_index(data_dir: str = "data/samples") -> Tuple[SemanticIndex, EmbeddingService, List[str], List[Dict[str, Any]]]:
    all_texts: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.lower().endswith((".pdf", ".docx", ".json")):
                continue
            fpath = os.path.join(root, fname)
            try:
                res = _read_any(fpath)
                if not res:
                    continue
                chunks = simple_normalize(res["chunks"])
                for ch in chunks:
                    text = ch["text"].strip()
                    if not text:
                        continue
                    meta = {
                        "filename": res["metadata"].get("filename"),
                        "format": res["metadata"].get("format", os.path.splitext(fname)[1].lstrip(".")),
                        **ch.get("meta", {}),
                        "path": fpath,
                    }
                    all_texts.append(text)
                    all_meta.append(meta)
            except Exception as e:
                print(f"[WARN] No se pudo procesar {fpath}: {e}")

    if not all_texts:
        raise RuntimeError(f"No se encontraron textos válidos en {data_dir}")

    emb = EmbeddingService()
    vecs = emb.embed(all_texts)

    index = SemanticIndex(dim=vecs.shape[1])
    index.add(vecs, all_meta)

    return index, emb, all_texts, all_meta
