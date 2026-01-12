"""
Unified document loader for loading and parsing multiple document formats.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rlm.parsers.docx_parser import parse_docx
from rlm.parsers.pdf_parser import parse_pdf
from rlm.parsers.pptx_parser import parse_pptx

# Supported file extensions and their parsers
PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".pptx": parse_pptx,
}

# Text file extensions (no special parsing needed)
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml", ".html"}

# Maximum file size (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Maximum total context size (1M characters)
MAX_TOTAL_SIZE = 1_000_000


class DocumentLoader:
    """Load and parse documents from a folder."""

    def __init__(self, folder_path: str | Path):
        self.folder_path = Path(folder_path)
        self.documents: List[Dict] = []
        self.errors: List[str] = []
        self.total_chars = 0

    def scan_folder(self) -> List[Path]:
        """Scan folder for supported files."""
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")

        supported_extensions = set(PARSERS.keys()) | TEXT_EXTENSIONS
        files = []

        for file_path in self.folder_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                # Skip hidden files and temp files
                if not file_path.name.startswith(".") and not file_path.name.startswith(
                    "~"
                ):
                    files.append(file_path)

        return sorted(files)

    def parse_file(
        self, file_path: Path, max_chars: int = None
    ) -> Tuple[str, Optional[str]]:
        """
        Parse a single file.

        Args:
            file_path: Path to the file
            max_chars: Maximum characters to read (for optimization)

        Returns:
            Tuple of (content, error_message or None)
        """
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return (
                "",
                f"File too large: {file_path.name} ({file_size / 1024 / 1024:.1f}MB)",
            )

        suffix = file_path.suffix.lower()

        try:
            if suffix in PARSERS:
                content = PARSERS[suffix](file_path)
            elif suffix in TEXT_EXTENSIONS:
                # Optimize: read only needed bytes for large text files
                read_bytes = max_chars * 4 if max_chars else None  # UTF-8 worst case

                for encoding in ["utf-8", "cp949", "euc-kr", "latin-1"]:
                    try:
                        if read_bytes and file_size > read_bytes:
                            # Read only what we need
                            with open(file_path, "r", encoding=encoding) as f:
                                content = f.read(max_chars + 1000)  # Read a bit extra
                        else:
                            content = file_path.read_text(encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return "", f"Failed to decode: {file_path.name}"
            else:
                return "", f"Unsupported format: {file_path.suffix}"

            return content, None
        except Exception as e:
            return "", f"Parse error in {file_path.name}: {e}"

    def load_all(self, max_chars: Optional[int] = None) -> str:
        """
        Load all documents from the folder and combine into context.

        Args:
            max_chars: Maximum total characters (default: MAX_TOTAL_SIZE)

        Returns:
            Combined context string with document separators
        """
        max_chars = max_chars or MAX_TOTAL_SIZE
        files = self.scan_folder()

        if not files:
            raise ValueError(f"No supported documents found in: {self.folder_path}")

        context_parts = []
        self.total_chars = 0

        for file_path in files:
            remaining_chars = max_chars - self.total_chars
            content, error = self.parse_file(file_path, max_chars=remaining_chars)

            if error:
                self.errors.append(error)
                continue

            if not content.strip():
                continue

            # Create document header
            rel_path = file_path.relative_to(self.folder_path)
            doc_header = f"\n{'='*80}\n📋 DOCUMENT: {rel_path}\n{'='*80}\n"
            doc_text = doc_header + content

            # Check size limit
            if self.total_chars + len(doc_text) > max_chars:
                remaining = max_chars - self.total_chars
                if remaining > 1000:  # Only add if meaningful portion fits
                    doc_text = doc_text[:remaining] + "\n...[TRUNCATED]"
                    context_parts.append(doc_text)
                    self.total_chars += len(doc_text)
                break

            context_parts.append(doc_text)
            self.total_chars += len(doc_text)

            self.documents.append(
                {
                    "path": str(rel_path),
                    "size": len(content),
                    "type": file_path.suffix.lower(),
                }
            )

        return "\n".join(context_parts)

    def get_summary(self) -> str:
        """Get a summary of loaded documents."""
        lines = [
            f"Loaded {len(self.documents)} documents ({self.total_chars:,} chars):"
        ]
        for doc in self.documents:
            lines.append(f"  - {doc['path']} ({doc['size']:,} chars)")
        if self.errors:
            lines.append(f"\nWarnings ({len(self.errors)}):")
            for err in self.errors[:5]:  # Show first 5 errors
                lines.append(f"  ⚠️ {err}")
        return "\n".join(lines)


def load_folder(
    folder_path: str | Path, max_chars: Optional[int] = None
) -> Tuple[str, str]:
    """
    Convenience function to load all documents from a folder.

    Args:
        folder_path: Path to folder containing documents
        max_chars: Maximum context size

    Returns:
        Tuple of (context_string, summary_string)
    """
    loader = DocumentLoader(folder_path)
    context = loader.load_all(max_chars)
    summary = loader.get_summary()
    return context, summary


def list_project_folders(base_path: str = "data/projects") -> List[Dict]:
    """
    List available project folders.

    Returns:
        List of dicts with folder info: {name, path, file_count}
    """
    base = Path(base_path)
    if not base.exists():
        return []

    folders = []
    supported_extensions = set(PARSERS.keys()) | TEXT_EXTENSIONS

    for folder in sorted(base.iterdir()):
        if folder.is_dir() and not folder.name.startswith("."):
            # Count supported files
            file_count = sum(
                1
                for f in folder.rglob("*")
                if f.is_file() and f.suffix.lower() in supported_extensions
            )
            if file_count > 0:
                folders.append(
                    {
                        "name": folder.name,
                        "path": str(folder),
                        "file_count": file_count,
                    }
                )

    return folders
