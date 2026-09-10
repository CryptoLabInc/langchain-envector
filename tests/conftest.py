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
    """Stand-in for ``pyenvector.Index``, mirroring the 1.6.x signatures.

    Every call is recorded so tests can assert on what the vector store asked
    the SDK to do. ``is_loaded`` starts False like a freshly created index, so
    the ``_loaded_index()`` guard is exercised.
    """

    inserted: List[Dict[str, Any]] = field(default_factory=list)
    deleted: List[Dict[str, Any]] = field(default_factory=list)
    updates: List[Dict[str, Any]] = field(default_factory=list)
    upserts: List[Dict[str, Any]] = field(default_factory=list)
    partitions: List[str] = field(default_factory=list)
    searched: List[Dict[str, Any]] = field(default_factory=list)
    stage_waits: List[Dict[str, Any]] = field(default_factory=list)
    search_payload: Optional[List[List[Dict[str, Any]]]] = None
    is_loaded: bool = False
    load_calls: int = 0
    next_item_id: int = 1
    row_count: int = 0

    def load(self):
        self.load_calls += 1
        self.is_loaded = True

    @property
    def indexer(self):
        return _FakeIndexer(self)

    def _issue_ids(self, count: int) -> List[int]:
        ids = list(range(self.next_item_id, self.next_item_id + count))
        self.next_item_id += count
        return ids

    def insert(
        self,
        data: List[List[float]],
        metadata: List[str],
        partition_name: Optional[str] = None,
        await_completion: bool = False,
        execute_until: str = "segmentation",
        load: bool = True,
        use_row_insert: bool = False,
        n_workers: int = 1,
        timeout_s: float = 86400.0,
        poll_interval_s: float = 1.0,
        request_ids: Optional[List[str]] = None,
    ) -> List[int]:
        self.inserted.append(
            {
                "data": data,
                "metadata": metadata,
                "partition_name": partition_name,
                "await_completion": await_completion,
                "execute_until": execute_until,
                "use_row_insert": use_row_insert,
                "n_workers": n_workers,
                "timeout_s": timeout_s,
                "poll_interval_s": poll_interval_s,
            }
        )
        if load:
            self.is_loaded = True
        if request_ids is not None:
            request_ids.append(f"req-ins-{len(self.inserted)}")
        self.row_count += len(metadata)
        return self._issue_ids(len(metadata))

    def wait_for_insert_stage(
        self,
        request_ids: List[str],
        target_stage: str,
        timeout_s: float = 600.0,
        poll_interval_s: float = 1.0,
        partition_name: Optional[str] = None,
    ) -> None:
        self.stage_waits.append(
            {
                "request_ids": list(request_ids),
                "target_stage": target_stage,
                "timeout_s": timeout_s,
                "partition_name": partition_name,
            }
        )

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
        self.row_count = max(0, self.row_count - len(item_ids))
        return f"req-del-{len(self.deleted)}"

    def update(
        self,
        items: List[Any],
        await_completion: bool = False,
        timeout_s: float = 600.0,
        poll_interval_s: float = 1.0,
        n_workers: int = 1,
        partition_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.updates.append(
            {
                "items": list(items),
                "await_completion": await_completion,
                "timeout_s": timeout_s,
                "poll_interval_s": poll_interval_s,
                "partition_name": partition_name,
            }
        )
        return {
            "request_id": f"req-upd-{len(self.updates)}",
            "not_found_item_ids": [],
        }

    def upsert(
        self,
        items: List[Any],
        await_completion: bool = False,
        timeout_s: float = 600.0,
        poll_interval_s: float = 1.0,
        n_workers: int = 1,
        partition_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.upserts.append(
            {
                "items": list(items),
                "await_completion": await_completion,
                "timeout_s": timeout_s,
                "poll_interval_s": poll_interval_s,
                "partition_name": partition_name,
            }
        )
        inserted = self._issue_ids(sum(1 for it in items if it.item_id is None))
        return {
            "request_id": f"req-ups-{len(self.upserts)}",
            "inserted_item_ids": inserted,
            "not_found_item_ids": [],
        }

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
        self.searched.append({"top_k": top_k, "partition_names": partition_names})
        if self.search_payload is not None:
            return self.search_payload
        # Default one-hit result with metadata JSON
        item = {
            "id": 1,
            "score": 0.9,
            "metadata": json.dumps({"text": "hello", "metadata": {"tag": "x"}}),
        }
        return [[item]]


class _FakeIndexer:
    """Minimal stand-in for the SDK Indexer, for the summary lookups we make."""

    def __init__(self, index: "FakeIndex"):
        self._index = index

    def get_index_summary(self, index_name: str):
        return {"index_name": index_name, "row_count": self._index.row_count}


class FakeClient:
    def __init__(self, index: Optional[FakeIndex] = None):
        self._index = index or FakeIndex()

    def init(self):
        return self

    @property
    def index(self):
        return self._index
