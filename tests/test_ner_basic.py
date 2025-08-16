from app.ml.ner_pipeline import NERService

def test_ner_loads():
    n = NERService()  # requiere en_core_web_sm instalado
    ents = n.analyze("Apple invested 200 USD in Brazil on 2025-08-14.")
    # No exigimos entidades exactas; solo que corre sin romperse
    assert isinstance(ents, list)
