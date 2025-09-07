from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class FakeEmbeddings:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Deterministic tiny vectors for testing
        return [[float(i % self.dim) for i, _ in enumerate(texts)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * self.dim


@dataclass
class FakeIndex:
    inserted: List[Dict[str, Any]] = field(default_factory=list)
    search_payload: Optional[List[List[Dict[str, Any]]]] = None

    def insert(self, data: List[List[float]], metadata: List[str]):
        self.inserted.append({"data": data, "metadata": metadata})
        return self

    def search(self, query: List[float], top_k: int, output_fields: List[str]):
        if self.search_payload is not None:
            return self.search_payload
        # Default one-hit result with metadata JSON
        item = {
            "id": "pos-0",
            "score": 0.9,
            "metadata": json.dumps({"text": "hello", "metadata": {"tag": "x"}}),
        }
        return [[item]]


class FakeClient:
    def __init__(self, index: Optional[FakeIndex] = None):
        self._index = index or FakeIndex()

    def init(self):
        return self

    @property
    def index(self):
        return self._index
