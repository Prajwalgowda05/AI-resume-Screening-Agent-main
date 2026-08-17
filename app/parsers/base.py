"""Base parser interface and custom exceptions for document parsing."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class DocumentParsingError(Exception):
    """Base exception for document parsing failures."""

    pass


class UnsupportedFileTypeError(DocumentParsingError):
    """Raised when an unsupported file format is encountered."""

    pass


class EmptyDocumentError(DocumentParsingError):
    """Raised when an extracted document contains no readable text."""

    pass


class BaseParser(ABC):
    """Abstract base class defining the parser contract."""

    @abstractmethod
    def parse(self, file_path: Union[str, Path]) -> str:
        """Parse a document file and return its extracted raw text content.

        Args:
            file_path: Path to the target document.

        Returns:
            Extracted text content as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            DocumentParsingError: If extraction fails.
        """
        pass

    def validate_file(self, file_path: Union[str, Path]) -> Path:
        """Validate that the file exists and is a regular file.

        Args:
            file_path: Path to the file.

        Returns:
            Resolved Path object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the path is a directory or invalid.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {file_path}")
        return path
