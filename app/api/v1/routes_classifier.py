from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from app.services.dataset import load_labeled_texts, _read_path_as_text
from app.ml.classifier import TextClassifier, MODEL_PATH

router = APIRouter()

_CSTATE = {
    "ready": False,
    "model": None,
    "labels": [],
    "model_path": MODEL_PATH,
    "train_dir": "data/train",
}

class TrainIn(BaseModel):
    train_dir: Optional[str] = None
    test_size: float = 0.2
    random_state: int = 42

class TrainOut(BaseModel):
    ok: bool
    train_dir: str
    classes: List[str]
    samples: int
    accuracy: float
    report: Dict[str, Any]

class ClassifyIn(BaseModel):
    text: Optional[str] = None
    path: Optional[str] = None
    topk: int = 3

class ClassifyOut(BaseModel):
    label: str
    probs: Dict[str, float]

@router.post("/classifier/train", response_model=TrainOut)
def train_classifier(body: TrainIn):
    train_dir = body.train_dir or _CSTATE["train_dir"]
    texts, labels = load_labeled_texts(train_dir=train_dir)
    Xtr, Xte, ytr, yte = train_test_split(
        texts, labels,
        test_size=body.test_size,
        random_state=body.random_state,
        stratify=labels if len(set(labels)) > 1 else None
    )

    clf = TextClassifier()
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    acc = float(accuracy_score(yte, ypred))
    try:
        rep = classification_report(yte, ypred, output_dict=True, zero_division=0)
    except Exception:
        rep = {}

    clf.save(MODEL_PATH)
    _CSTATE.update({"ready": True, "model": clf, "labels": list(sorted(set(labels))), "train_dir": train_dir})

    return TrainOut(
        ok=True,
        train_dir=train_dir,
        classes=sorted(list(set(labels))),
        samples=len(texts),
        accuracy=acc,
        report=rep
    )

@router.post("/classifier", response_model=ClassifyOut)
def classify(body: ClassifyIn):
    if not _CSTATE["ready"]:
        # Intentar cargar desde disco
        try:
            clf = TextClassifier().load(MODEL_PATH)
            _CSTATE.update({"ready": True, "model": clf})
        except Exception:
            raise HTTPException(status_code=400, detail="Modelo no entrenado. Ejecuta POST /v1/classifier/train")

    clf: TextClassifier = _CSTATE["model"]

    if body.text:
        text = body.text
    elif body.path:
        text = _read_path_as_text(body.path)
        if not text.strip():
            raise HTTPException(status_code=400, detail=f"No se pudo leer contenido de {body.path}")
    else:
        raise HTTPException(status_code=422, detail="Debes enviar 'text' o 'path'")

    probs = clf.predict_proba([text])[0]
    if probs:
        # Ordenar y elegir top-1
        label = max(probs.items(), key=lambda kv: kv[1])[0]
        # Limitar a topk
        topk = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[: max(1, body.topk)]
        probs = {k: float(v) for k, v in topk}
    else:
        label = clf.predict([text])[0]
        probs = {label: 1.0}

    return ClassifyOut(label=label, probs=probs)
