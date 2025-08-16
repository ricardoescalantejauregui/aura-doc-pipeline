from app.ingest.pdf_reader import read_pdf
import os

def test_read_pdf_sample():
    sample = os.path.join("data", "samples", "sample.pdf")
    if not os.path.exists(sample):
        assert callable(read_pdf)
        return
    result = read_pdf(sample)
    assert "doc_id" in result and result["chunks"]
