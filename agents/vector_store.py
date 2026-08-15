from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from agents.chunking import DocumentChunk
from agents.embeddings import EmbeddingResult


@dataclass
class VectorStore:
    embeddings: np.ndarray
    chunks: List[DocumentChunk]
    vocabulary: Dict[str, int]
    idf: np.ndarray

    @property
    def ntotal(self) -> int:
        return int(self.embeddings.shape[0])


class VectorStoreAgent:
    def build_or_load(self, document_key: str, chunks: List[DocumentChunk], embeddings: EmbeddingResult) -> VectorStore:
        if len(chunks) == 0 or embeddings.matrix.size == 0:
            raise ValueError("No searchable text was extracted from this source.")

        return VectorStore(
            embeddings=embeddings.matrix,
            chunks=chunks,
            vocabulary=embeddings.vocabulary,
            idf=embeddings.idf,
        )
