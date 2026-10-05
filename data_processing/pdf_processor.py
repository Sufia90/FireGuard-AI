"""PDF text extraction for the RAG pipeline (via PyMuPDF).

Used by rag/document_ingestion.py to turn NASA PDFs into page-tagged text.
Table extraction is intentionally left as a documented extension point —
see extract_tables_from_pdf().
"""
from pathlib import Path


def _get_fitz():
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is not installed. Run: pip install pymupdf"
        ) from exc
    return fitz


def extract_pages_from_pdf(path):
    """Return [(page_number, text), ...] for every page of the PDF."""
    fitz = _get_fitz()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            try:
                text = page.get_text() or ""
            except Exception as exc:
                print(f"Warning: could not extract text from page {i + 1} "
                      f"of {path.name}: {exc}")
                text = ""
            pages.append((i + 1, text))
    return pages


def extract_text_from_pdf(path):
    """Return the whole document as one string (pages joined)."""
    pages = extract_pages_from_pdf(path)
    full_text = "\n\n".join(text for _, text in pages).strip()
    if not full_text:
        print(f"Note: no selectable text found in {Path(path).name}. "
              "It may be a scanned PDF — try ocr_processor.ocr_image().")
    return full_text


def extract_tables_from_pdf(path):
    raise NotImplementedError(
        "Table extraction is a planned extension, not part of the starter. "
        "PyMuPDF can do this natively: page.find_tables() -> table.extract()."
    )
