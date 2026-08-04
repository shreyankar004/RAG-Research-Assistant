"""
Tiny TF-IDF embedding agent for memory-constrained deployments.

This avoids FastEmbed/ONNX model loading and keeps Railway/Render free-tier
memory usage predictable. Each vector store keeps its own vocabulary so PDF
and web research can be queried independently.
"""
from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Dict, Iterable, List

import numpy as np

from config import TFIDF_MAX_FEATURES


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]{1,}")


@dataclass
class EmbeddingResult:
    matrix: np.ndarray
    vocabulary: Dict[str, int]
    idf: np.ndarray


class EmbeddingAgent:
    def __init__(self, max_features: int = TFIDF_MAX_FEATURES):
        self.max_features = max_features

    def _tokens(self, text: str) -> List[str]:
        return [token.lower() for token in TOKEN_RE.findall(text)]

    def embed_documents(self, texts: Iterable[str]) -> EmbeddingResult:
        items = list(texts)
        if not items:
            return EmbeddingResult(
                matrix=np.empty((0, 0), dtype="float32"),
                vocabulary={},
                idf=np.empty((0,), dtype="float32"),
            )

        tokenized = [self._tokens(text) for text in items]
        document_frequency: Counter[str] = Counter()
        corpus_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            counts = Counter(tokens)
            corpus_frequency.update(counts)
            document_frequency.update(counts.keys())

        terms = [
            term
            for term, _ in corpus_frequency.most_common(self.max_features)
            if document_frequency[term] > 0
        ]
        vocabulary = {term: index for index, term in enumerate(terms)}
        if not vocabulary:
            return EmbeddingResult(
                matrix=np.empty((len(items), 0), dtype="float32"),
                vocabulary={},
                idf=np.empty((0,), dtype="float32"),
            )

        doc_count = len(items)
        idf = np.array(
            [math.log((1 + doc_count) / (1 + document_frequency[term])) + 1 for term in terms],
            dtype="float32",
        )
        matrix = np.zeros((doc_count, len(vocabulary)), dtype="float32")

        for row, tokens in enumerate(tokenized):
            counts = Counter(token for token in tokens if token in vocabulary)
            for token, count in counts.items():
                matrix[row, vocabulary[token]] = math.log1p(count)

        matrix *= idf
        return EmbeddingResult(matrix=self._normalize(matrix), vocabulary=vocabulary, idf=idf)

    def embed_query(self, text: str, vocabulary: Dict[str, int], idf: np.ndarray) -> np.ndarray:
        if not vocabulary:
            return np.empty((1, 0), dtype="float32")

        vector = np.zeros((1, len(vocabulary)), dtype="float32")
        counts = Counter(token for token in self._tokens(text) if token in vocabulary)
        for token, count in counts.items():
            vector[0, vocabulary[token]] = math.log1p(count)
        vector *= idf
        return self._normalize(vector)

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix.astype("float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (matrix / norms).astype("float32")
