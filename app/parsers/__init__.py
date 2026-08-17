"""Document parsing package providing a unified interface for PDF, DOCX, and TXT files."""

from pathlib import Path
from typing import Dict, Type, Union

from app.parsers.base import (
    BaseParser,
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from app.parsers.docx_parser import DocxParser
from app.parsers.pdf_parser import PDFParser
from app.parsers.text_parser import TextParser

# Mapping from file extension (lowercase) to parser class
PARSER_REGISTRY: Dict[str, Type[BaseParser]] = {
    ".pdf": PDFParser,
    ".docx": DocxParser,
    ".txt": TextParser,
}


def get_parser(file_path: Union[str, Path]) -> BaseParser:
    """Retrieve the appropriate document parser for a given file.

    Args:
        file_path: Path to the target document.

    Returns:
        An instance of BaseParser capable of parsing the document.

    Raises:
        UnsupportedFileTypeError: If the file extension is not supported.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    parser_cls = PARSER_REGISTRY.get(extension)
    if not parser_cls:
        supported = ", ".join(PARSER_REGISTRY.keys())
        raise UnsupportedFileTypeError(
            f"Unsupported file format '{extension}' for file '{path.name}'. "
            f"Supported formats: {supported}"
        )

    return parser_cls()


def parse_document(file_path: Union[str, Path]) -> str:
    """Parse a document file (PDF, DOCX, or TXT) and return its extracted raw text.

    Args:
        file_path: Path to the file.

    Returns:
        Extracted raw text content.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnsupportedFileTypeError: If the file extension is not supported.
        DocumentParsingError: If extraction fails.
    """
    parser = get_parser(file_path)
    return parser.parse(file_path)


__all__ = [
    "BaseParser",
    "PDFParser",
    "DocxParser",
    "TextParser",
    "DocumentParsingError",
    "UnsupportedFileTypeError",
    "EmptyDocumentError",
    "get_parser",
    "parse_document",
    "PARSER_REGISTRY",
]
