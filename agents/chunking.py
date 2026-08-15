from dataclasses import dataclass
from typing import List

from config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass
class DocumentChunk:
    id: int
    text: str
    start: int
    end: int


class ChunkingAgent:
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[DocumentChunk]:
        clean_text = " ".join(text.split())
        if not clean_text:
            return []

        chunks = []
        start = 0
        chunk_id = 0
        while start < len(clean_text):
            end = min(start + self.chunk_size, len(clean_text))
            if end < len(clean_text):
                boundary = max(
                    clean_text.rfind(".", start, end),
                    clean_text.rfind("\n", start, end),
                    clean_text.rfind(" ", start, end),
                )
                if boundary > start + self.chunk_size // 2:
                    end = boundary + 1

            chunk_text = clean_text[start:end].strip()
            if chunk_text:
                chunks.append(DocumentChunk(id=chunk_id, text=chunk_text, start=start, end=end))
                chunk_id += 1

            if end >= len(clean_text):
                break
            start = max(0, end - self.overlap)

        return chunks
