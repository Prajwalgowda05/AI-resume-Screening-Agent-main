"""PDF document parser implementation using pypdf."""

from pathlib import Path
from typing import Union
import pypdf

from app.parsers.base import BaseParser, DocumentParsingError


class PDFParser(BaseParser):
    """Parser for extracting text from PDF files using pypdf."""

    def parse(self, file_path: Union[str, Path]) -> str:
        """Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text from all pages.

        Raises:
            FileNotFoundError: If the file does not exist.
            DocumentParsingError: If PDF reading or extraction fails.
        """
        path = self.validate_file(file_path)

        try:
            reader = pypdf.PdfReader(str(path))
            pages_text = []
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text.strip())

            extracted_text = "\n\n".join(pages_text).strip()
            return extracted_text
        except FileNotFoundError:
            raise
        except Exception as e:
            raise DocumentParsingError(
                f"Failed to extract text from PDF '{path.name}': {str(e)}"
            ) from e
