"""OCR for scanned documents (optional — needs Tesseract installed).

Only used when a document has no selectable text (e.g. scanned pages).
Everything else in FireGuard AI works without it.
"""
from pathlib import Path


def ocr_image(path: str | Path, lang: str = "eng") -> str:
    """Extract text from an image file via Tesseract OCR."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is not installed, so OCR is unavailable.\n"
            "To enable it:\n"
            "  1. pip install pytesseract\n"
            "  2. Install the Tesseract-OCR engine for your OS\n"
            "     (e.g. https://tesseract-ocr.github.io/tessdoc/Installation.html)\n"
            "OCR is optional — everything else in FireGuard AI works without it."
        ) from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for OCR but is not installed. "
            "Run: pip install Pillow"
        ) from exc

    try:
        with Image.open(path) as img:
            return pytesseract.image_to_string(img, lang=lang)
    except Exception as exc:
        raise RuntimeError(
            f"OCR failed on {path.name}: {exc}\n"
            "If pytesseract is installed, the Tesseract-OCR engine itself "
            "may be missing from your system PATH."
        ) from exc

