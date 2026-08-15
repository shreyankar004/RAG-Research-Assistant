from dataclasses import dataclass
from typing import List

import numpy as np

from agents.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    text: str
    score: float
    chunk_id: int


class RetrieverAgent:
    def retrieve(self, vector_store: VectorStore, query_embedding: np.ndarray, top_k: int = 4) -> List[RetrievedChunk]:
        k = min(top_k, vector_store.ntotal)
        if k <= 0:
            return []

        scores = (vector_store.embeddings @ query_embedding.T).ravel()
        indexes = np.argsort(scores)[::-1][:k]
        results = []
        for idx in indexes:
            score = scores[int(idx)]
            chunk = vector_store.chunks[int(idx)]
            results.append(RetrievedChunk(text=chunk.text, score=float(score), chunk_id=chunk.id))
        return results
