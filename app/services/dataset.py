import os, json
from typing import List, Tuple
from app.ingest.normalizer import simple_normalize
from app.services.pipeline import _read_any  # reutilizamos lectores de pdf/docx/json

def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_path_as_text(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".txt"):
        return _read_txt(path)
    res = _read_any(path)
    if not res:
        return ""
    chunks = simple_normalize(res["chunks"])
    return "\n".join([c["text"] for c in chunks if c["text"].strip()])

def load_labeled_texts(train_dir: str = "data/train") -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    for label in sorted(os.listdir(train_dir)):
        lbl_path = os.path.join(train_dir, label)
        if not os.path.isdir(lbl_path):
            continue
        for root, _, files in os.walk(lbl_path):
            for fname in files:
                if not fname.lower().endswith((".txt", ".json", ".pdf", ".docx")):
                    continue
                path = os.path.join(root, fname)
                try:
                    text = _read_path_as_text(path)
                    if text.strip():
                        texts.append(text)
                        labels.append(label)
                except Exception as e:
                    print(f"[WARN] No se pudo leer {path}: {e}")
    if not texts:
        raise RuntimeError(f"No se encontraron ejemplos en {train_dir}")
    return texts, labels
