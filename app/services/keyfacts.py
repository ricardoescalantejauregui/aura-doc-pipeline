import re
from typing import Dict, Any, List, Optional

_MONEY_RX = re.compile(r"(?i)(?:\$?\s?\d[\d,.\s]*\s?(?:USD|EUR|MXN|\$)?)")
_DATE_ISO_RX = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")          # 2025-08-14
_DATE_SLASH_RX = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")  # 14/08/2025, 14-08-25
_ID_RX = re.compile(r"(?i)\b(?:invoice|factura|folio|id|no\.?)\s*[:#]?\s*([A-Za-z0-9_-]{4,})")

def extract_key_facts(text: str, ents: Optional[List[dict]] = None) -> Dict[str, Any]:
    """
    Extrae datos clave combinando entidades NER y regex simples.
    Devuelve listas para mantener múltiples coincidencias.
    """
    facts = {
        "money": [],
        "dates": [],
        "ids": [],
        "orgs": [],
        "persons": [],
        "locations": [],
    }
    t = text or ""

    # 1) Por entidades (si se proporcionan)
    if ents:
        for e in ents:
            label = e.get("label")
            val = (e.get("text") or "").strip()
            if not val:
                continue
            if label == "MONEY":
                facts["money"].append(val)
            elif label in ("DATE",):
                facts["dates"].append(val)
            elif label in ("ORG",):
                facts["orgs"].append(val)
            elif label in ("PERSON",):
                facts["persons"].append(val)
            elif label in ("GPE", "LOC"):
                facts["locations"].append(val)

    # 2) Por regex (complementa entidades)
    facts["money"].extend(_MONEY_RX.findall(t))
    facts["dates"].extend(_DATE_ISO_RX.findall(t))
    facts["dates"].extend(_DATE_SLASH_RX.findall(t))
    facts["ids"].extend([m.group(1) for m in _ID_RX.finditer(t)])

    # Normalización simple (dedup + limpiar espacios)
    for k in facts:
        dedup = []
        seen = set()
        for v in facts[k]:
            vv = " ".join(v.split())
            if vv and vv not in seen:
                dedup.append(vv)
                seen.add(vv)
        facts[k] = dedup

    return facts
