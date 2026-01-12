"""
Document parsers module for extracting text from various file formats.
Supports PDF, DOCX, PPTX, TXT, and MD files.
"""

from rlm.parsers.loader import DocumentLoader, load_folder

__all__ = ["DocumentLoader", "load_folder"]
