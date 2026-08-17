"""Plain text document parser implementation."""

from pathlib import Path
from typing import Union

from app.parsers.base import BaseParser, DocumentParsingError


class TextParser(BaseParser):
    """Parser for extracting text from plain text files."""

    def parse(self, file_path: Union[str, Path]) -> str:
        """Extract text from a plain text (.txt) file.

        Args:
            file_path: Path to the text file.

        Returns:
            Extracted text content.

        Raises:
            FileNotFoundError: If the file does not exist.
            DocumentParsingError: If reading the file fails.
        """
        path = self.validate_file(file_path)

        # Attempt decoding with UTF-8 first, then fallback encodings
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        last_error = None

        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as f:
                    return f.read().strip()
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                raise DocumentParsingError(
                    f"Failed to read text file '{path.name}': {str(e)}"
                ) from e

        raise DocumentParsingError(
            f"Failed to decode text file '{path.name}' with supported encodings: {str(last_error)}"
        )
