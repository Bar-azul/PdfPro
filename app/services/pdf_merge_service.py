from pathlib import Path
from typing import List
from pypdf import PdfReader, PdfWriter


class PDFMergeService:
    @staticmethod
    def merge_pdfs(input_files: List[Path], output_file: Path) -> Path:
        """
        Merge multiple PDF files into one PDF file.
        """

        if not input_files:
            raise ValueError("No PDF files were provided.")

        writer = PdfWriter()

        for file_path in input_files:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if file_path.suffix.lower() != ".pdf":
                raise ValueError(f"Invalid file type: {file_path.name}")

            reader = PdfReader(str(file_path))

            for page in reader.pages:
                writer.add_page(page)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "wb") as f:
            writer.write(f)

        return output_file