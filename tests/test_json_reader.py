from app.ingest.json_reader import read_json
import os, json

def test_read_json_sample(tmp_path):
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    result = read_json(str(sample))
    assert "doc_id" in result and result["chunks"]
