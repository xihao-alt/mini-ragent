"""Load text from local PDF documents."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def load_pdf(path: str | Path) -> str:
    """Return the extractable text from every page of a PDF file."""
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages = [text.strip() for page in reader.pages if (text := page.extract_text())]
    return "\n\n".join(pages)
