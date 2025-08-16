from typing import List, Dict, Any
import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "data/model/classifier.joblib" #Parametrizar variables de ambiente

class TextClassifier:
    def __init__(self):
        self.pipeline: Pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                n_jobs=None,
            ))
        ])
        self.labels_: List[str] = []

    def fit(self, texts: List[str], labels: List[str]):
        self.labels_ = sorted(list(set(labels)))
        self.pipeline.fit(texts, labels)

    def predict(self, texts: List[str]) -> List[str]:
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            proba = self.pipeline.predict_proba(texts)
            classes = list(self.pipeline.named_steps["clf"].classes_)
            out: List[Dict[str, float]] = []
            for row in proba:
                out.append({cls: float(p) for cls, p in zip(classes, row)})
            return out
        # Fallback: usar decision_function si no hay probas
        if hasattr(self.pipeline.named_steps["clf"], "decision_function"):
            scores = self.pipeline.named_steps["clf"].decision_function(texts)
            if scores.ndim == 1:  # binario
                scores = np.vstack([-scores, scores]).T
            exp = np.exp(scores - scores.max(axis=1, keepdims=True))
            proba = exp / exp.sum(axis=1, keepdims=True)
            classes = list(self.pipeline.named_steps["clf"].classes_)
            out = []
            for row in proba:
                out.append({cls: float(p) for cls, p in zip(classes, row)})
            return out
        return [{} for _ in texts]

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.pipeline}, path)

    def load(self, path: str = MODEL_PATH):
        data = joblib.load(path)
        self.pipeline = data["model"]
        return self
