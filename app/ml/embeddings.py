from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)
