from typing import List, Dict, Any
import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

class SemanticIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._meta: List[Dict[str, Any]] = []
        self._faiss = faiss.IndexFlatIP(dim) if _HAS_FAISS else None
        self._embs = None  # fallback NumPy

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        if vectors.ndim != 2:
            raise ValueError("vectors debe ser [n_items, dim]")
        if len(metadata) != vectors.shape[0]:
            raise ValueError("metadata y vectors deben tener mismo tamaño")
        self._meta.extend(metadata)
        if self._faiss:
            self._faiss.add(vectors.astype(np.float32))
        else:
            self._embs = vectors if self._embs is None else np.vstack([self._embs, vectors])

    def search(self, query_vecs: np.ndarray, k: int = 5):
        if self._faiss:
            D, I = self._faiss.search(query_vecs.astype(np.float32), k)
        else:
            if self._embs is None:
                return [[] for _ in range(query_vecs.shape[0])]
            sims = query_vecs @ self._embs.T  # coseno (ya normalizados)
            I = np.argsort(-sims, axis=1)[:, :k]
            D = np.take_along_axis(sims, I, axis=1)

        results = []
        for row in range(I.shape[0]):
            hits = []
            for idx, score in zip(I[row], D[row]):
                if idx < 0:
                    continue
                hits.append({"index": int(idx), "score": float(score), "meta": self._meta[int(idx)]})
            results.append(hits)
        return results
