"""Document loading + chunking for the RAG knowledge base.

- chunk_text(): word-based chunks with overlap.
- load_documents(): plain text loader (.md/.txt/.pdf).
- load_documents_with_pages(): keeps PDF page numbers so evidence can cite
  "SRD_BASS-II.pdf — page 7".
"""
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def chunk_text(text, chunk_size=400, overlap=50):
    """Split text into overlapping word chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def load_documents(folder):
    """Load supported files from a folder as [{source, text}]."""
    folder = Path(folder)
    if not folder.exists():
        print(f"Document folder not found: {folder} — nothing to ingest.")
        return []

    documents = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            if path.suffix.lower() == ".pdf":
                from data_processing.pdf_processor import extract_text_from_pdf
                text = extract_text_from_pdf(path)
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
            continue
        if text.strip():
            documents.append({"source": path.name, "text": text})
        else:
            print(f"Skipping {path.name}: no extractable text.")
    print(f"Loaded {len(documents)} document(s) from {folder}")
    return documents


def load_documents_with_pages(folder):
    """Like load_documents(), but PDFs keep per-page text for citations."""
    folder = Path(folder)
    if not folder.exists():
        print(f"Document folder not found: {folder} — nothing to ingest.")
        return []

    documents = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            if path.suffix.lower() == ".pdf":
                from data_processing.pdf_processor import extract_pages_from_pdf
                page_texts = extract_pages_from_pdf(path)
                text = "\n\n".join(t for _, t in page_texts)
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
                page_texts = None
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
            continue
        if text.strip():
            documents.append({"source": path.name, "text": text,
                              "page_texts": page_texts})
        else:
            print(f"Skipping {path.name}: no extractable text.")
    print(f"Loaded {len(documents)} document(s) from {folder}")
    return documents
