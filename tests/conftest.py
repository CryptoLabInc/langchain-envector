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
    deleted: List[Dict[str, Any]] = field(default_factory=list)
    metadata_updates: List[Dict[str, Any]] = field(default_factory=list)
    partitions: List[str] = field(default_factory=list)
    searched: List[Dict[str, Any]] = field(default_factory=list)
    search_payload: Optional[List[List[Dict[str, Any]]]] = None

    def insert(
        self,
        data: List[List[float]],
        metadata: List[str],
        partition_name: Optional[str] = None,
    ):
        self.inserted.append(
            {"data": data, "metadata": metadata, "partition_name": partition_name}
        )
        return [len(self.inserted) + i + 1 for i in range(len(metadata))]

    def delete(
        self,
        item_ids: List[int],
        await_completion: bool = True,
        timeout_s: float = 600.0,
        poll_interval_s: float = 1.0,
        partition_name: Optional[str] = None,
    ) -> str:
        self.deleted.append(
            {
                "item_ids": list(item_ids),
                "await_completion": await_completion,
                "timeout_s": timeout_s,
                "poll_interval_s": poll_interval_s,
                "partition_name": partition_name,
            }
        )
        return f"req-del-{len(self.deleted)}"

    def update_metadata(
        self,
        item_ids: List[int],
        metadata: List[Any],
        partition_name: Optional[str] = None,
    ) -> Dict[str, List[int]]:
        self.metadata_updates.append(
            {
                "item_ids": list(item_ids),
                "metadata": list(metadata),
                "partition_name": partition_name,
            }
        )
        return {"updated": list(item_ids), "skipped": []}

    def create_partition(self, partition_name: str):
        self.partitions.append(partition_name)
        return partition_name

    def drop_partition(self, partition_name: str):
        self.partitions.remove(partition_name)
        return partition_name

    def list_partitions(self):
        return [
            {"name": n, "status": "READY", "num_vectors": 0} for n in self.partitions
        ]

    def search(
        self,
        query: List[float],
        top_k: int,
        output_fields: List[str],
        partition_names: Optional[List[str]] = None,
    ):
        self.searched.append(
            {"top_k": top_k, "partition_names": partition_names}
        )
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
