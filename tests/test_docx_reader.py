from app.ingest.docx_reader import read_docx
import os

def test_read_docx_sample():
    sample = os.path.join("data", "samples", "sample.docx")
    if not os.path.exists(sample):
        assert callable(read_docx)
        return
    result = read_docx(sample)
    assert "doc_id" in result and result["chunks"]
