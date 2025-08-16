from typing import List, Dict, Any
import spacy

class NERService:
    """
    Servicio simple de NER con spaCy.
    Usa en_core_web_sm por defecto (entidades: PERSON, ORG, GPE, DATE, MONEY, etc.).
    """
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError as e:
            raise RuntimeError(
                f"spaCy model '{model_name}' no está instalado. "
                f"Instala con: python -m spacy download {model_name}"
            ) from e

    def analyze(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text or "")
        ents: List[Dict[str, Any]] = []
        for ent in doc.ents:
            ents.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
        return ents
