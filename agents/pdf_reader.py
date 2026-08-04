from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

from pypdf import PdfReader


@dataclass
class PDFReadResult:
    text: str
    page_count: int
    document_key: str


class PDFReaderAgent:
    def extract(self, uploaded_file) -> PDFReadResult:
        file_bytes = uploaded_file.getvalue()
        document_key = sha256(file_bytes).hexdigest()
        reader = PdfReader(BytesIO(file_bytes))

        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"\n\n[Page {page_number}]\n{text.strip()}")

        return PDFReadResult(
            text="\n".join(pages).strip(),
            page_count=len(reader.pages),
            document_key=document_key,
        )
