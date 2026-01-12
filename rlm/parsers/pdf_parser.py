"""PDF parser using pymupdf (fitz) - much faster than pypdf."""

from pathlib import Path

try:
    import pymupdf  # New import style
except ImportError:
    import fitz as pymupdf  # Fallback


def parse_pdf(file_path: Path) -> str:
    """
    Extract text from a PDF file using pymupdf (fast C-based parser).

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text content
    """
    try:
        doc = pymupdf.open(file_path)
        text_parts = []

        for i, page in enumerate(doc, 1):
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(f"[Page {i}]\n{page_text}")

        doc.close()
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"[PDF Parse Error: {e}]"
