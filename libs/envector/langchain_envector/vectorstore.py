from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from .config import EnvectorConfig
from .client import EnvectorClient
from .types import Embeddings, as_embeddings, pack_metadata, unpack_metadata

# pyenvector caps a single update/upsert call at 10_000 items
# (pyenvector.index.index.MAX_MUTATION_ITEMS_PER_CALL). Larger requests are
# split here so callers do not have to chunk by hand.
MAX_MUTATION_ITEMS_PER_CALL = 10000


def _mutation_items(item_ids: List[Any], label: str) -> List[int]:
    """Coerce caller-supplied IDs to the ``int`` item_ids the SDK addresses."""
    try:
        return [int(x) for x in item_ids]
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Envector.{label} expects integer item IDs (or numeric strings) "
            "as returned by add_texts/add_documents."
        ) from e


def _chunked(items: List[Any], size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _is_empty_shard_list_error(exc: Exception) -> bool:
    """True for the backend's "index has no shards" answer to a search.

    Matched on the message rather than the type: the SDK raises a generic
    ``InternalError`` wrapping the backend's gRPC NotFound, so the type alone
    would also swallow unrelated server failures. The message alone is not
    enough to act on either — see `_index_is_empty`.
    """
    message = str(exc)
    return "shard list for index" in message and "is empty" in message


def _try_import_item_types():
    """Return (UpdateItem, UpsertItem), falling back to structural stand-ins.

    The SDK reads these by attribute (`item_id` / `vector` / `metadata`), so the
    stand-ins are wire-compatible. They exist only so the unit tests, which run
    against fakes, keep working without the SDK installed — the same reason
    `_try_import_langchain` shims `Document`.
    """
    try:
        from pyenvector import UpdateItem, UpsertItem  # type: ignore

        return UpdateItem, UpsertItem
    except Exception:  # pragma: no cover - exercised only without the SDK

        @dataclass
        class UpdateItem:  # type: ignore[no-redef]
            item_id: int
            vector: Optional[List[float]] = None
            metadata: Optional[Any] = None

        @dataclass
        class UpsertItem:  # type: ignore[no-redef]
            item_id: Optional[int] = None
            vector: Optional[List[float]] = None
            metadata: Optional[Any] = None

        return UpdateItem, UpsertItem


def _try_import_langchain():
    """Return (VectorStoreBase, DocumentClass) with safe fallbacks.

    Ensures we always return a valid base class even if LangChain is missing.
    """
    VectorStoreBase: Any = object

    try:
        from langchain_core.documents import Document  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        # Minimal shim if LangChain is not installed
        class Document:  # type: ignore
            def __init__(
                self, page_content: str, metadata: Optional[Dict[str, Any]] = None
            ):
                self.page_content = page_content
                self.metadata = metadata or {}

    try:
        from langchain_core.vectorstores import VectorStore as _VectorStore  # type: ignore

        VectorStoreBase = _VectorStore
    except Exception:  # pragma: no cover - optional dependency
        pass

    return VectorStoreBase, Document


VectorStore, Document = _try_import_langchain()


class Envector(VectorStore):  # type: ignore[misc]
    """LangChain-compatible VectorStore adaptor for Envector.

    This class wraps the high-level `pyenvector` SDK. It does not use low-level
    gRPC stubs or `pyenvector.api.Indexer` directly.
    """

    def __init__(
        self,
        *,
        config: EnvectorConfig,
        embeddings: Optional[Embeddings] = None,
        client: Optional[EnvectorClient] = None,
    ) -> None:
        self.config = config
        self._embeddings = as_embeddings(embeddings) if embeddings is not None else None
        self.client = client or EnvectorClient(config)
        self.client.init()
        # Insert request ids whose server-side merge has not been waited for,
        # keyed by partition. See `_drain_pending_inserts`.
        self._pending_inserts: Dict[Optional[str], List[str]] = {}

    def _index_is_empty(self) -> bool:
        """Ask the server whether this index currently holds no rows.

        Used to decide whether an "index has no shards" search error means the
        index is genuinely empty. Any failure to answer counts as "not known to
        be empty", so the original error is re-raised rather than swallowed.
        """
        try:
            summary = self.client.index.indexer.get_index_summary(
                self.config.index.index_name
            )
            return int(summary["row_count"]) == 0
        except Exception:
            return False

    def _loaded_index(self):
        """Return the bound Index, loading it first if the server has not.

        Fresh indexes start unloaded, and pyenvector raises
        ``ValueError("Index not loaded")`` from search, delete, update and
        upsert rather than loading implicitly. This is not new in 1.6 — the
        guard is in 1.5 too; the search path just never went through here.
        """
        index = self.client.index
        if not getattr(index, "is_loaded", True):
            index.load()
        return index

    # -------------------------------
    # VectorStore API
    # -------------------------------
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        *,
        vectors: Optional[List[List[float]]] = None,
        partition_name: Optional[str] = None,
        await_completion: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[int]:
        """Add texts to the encrypted index and return their item IDs.

        If embeddings are provided, the texts are embedded automatically.
        Otherwise, provide pre-computed `vectors`. Pass `partition_name` to
        insert into a named partition.

        Inserted rows are searchable straight away: `Index.insert` publishes them
        via its own ``load`` step, so a following `similarity_search` sees them
        without any wait here. ``await_completion=True`` additionally blocks
        until the shards are merged and saved — durability rather than
        visibility — which is off by default (``config.write.await_insert``).

        Any other keyword argument goes straight to ``Index.insert``, which is
        where the SDK's own tuning knobs live (``execute_until``, ``n_workers``,
        ``use_row_insert``, ...).

        Notes:
        - Manual `ids` are ignored: enVector issues its own item IDs. Use the
          returned IDs with `delete` / `update_documents` / `upsert_documents`.
        """
        if not texts:
            return []

        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError("texts and metadatas must have equal length")

        if vectors is None:
            if self._embeddings is None:
                raise ValueError("embeddings is None and vectors not provided")
            vectors = self._embeddings.embed_documents(texts)

        # Prepare metadata JSON strings per item
        packed = [pack_metadata(t, m) for t, m in zip(texts, metadatas)]

        w = self.config.write
        awaited = w.await_insert if await_completion is None else await_completion
        request_ids: List[str] = kwargs.pop("request_ids", [])
        item_ids = self.client.index.insert(
            data=vectors,
            metadata=packed,
            partition_name=partition_name,
            request_ids=request_ids,
            await_completion=awaited,
            timeout_s=kwargs.pop("timeout_s", w.timeout_s),
            poll_interval_s=kwargs.pop("poll_interval_s", w.poll_interval_s),
            **kwargs,
        )
        if not awaited and request_ids:
            self._pending_inserts.setdefault(partition_name, []).extend(request_ids)
        return item_ids

    def delete(
        self,
        ids: Optional[List[Any]] = None,
        *,
        await_completion: Optional[bool] = None,
        timeout_s: Optional[float] = None,
        poll_interval_s: Optional[float] = None,
        partition_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete items from the encrypted index by item ID.

        Accepts the ``item_id`` values returned from ``add_texts`` /
        ``add_documents``. Both ``int`` and ``str`` (numeric) IDs are accepted
        and coerced to ``int`` before being passed to the SDK.

        Deletion is asynchronous server-side; by default this waits until the
        affected shards are rebuilt and the remaining data is searchable again
        (``config.write.await_delete``). IDs matching no live row are a no-op.
        """
        if not ids:
            return False
        item_ids = _mutation_items(list(ids), "delete")

        w = self.config.write
        self._loaded_index().delete(
            item_ids=item_ids,
            await_completion=(
                w.await_delete if await_completion is None else await_completion
            ),
            timeout_s=w.timeout_s if timeout_s is None else timeout_s,
            poll_interval_s=(
                w.poll_interval_s if poll_interval_s is None else poll_interval_s
            ),
            partition_name=partition_name,
        )
        return True

    # -------------------------------
    # In-place mutation (pyenvector >= 1.6.0)
    # -------------------------------
    def update_metadata(
        self,
        ids: List[Any],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        *,
        partition_name: Optional[str] = None,
        await_completion: Optional[bool] = None,
        timeout_s: Optional[float] = None,
        poll_interval_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Replace the stored text/metadata of existing items by item ID.

        Each item's stored payload is replaced wholesale with the packed
        ``{"text": ..., "metadata": ...}`` envelope built from ``texts[i]`` /
        ``metadatas[i]`` — supply the full new content, not a partial patch.
        Vectors are left untouched, so an item keeps matching its original
        embedding; use `update_documents` to replace the vector as well.

        Returns the merged SDK result: ``{"request_id": [...],
        "not_found_item_ids": [...]}``. ``not_found_item_ids`` lists IDs that
        matched no live row (missing or already deleted) — those are reported,
        not raised.
        """
        if not ids:
            return {"request_id": [], "not_found_item_ids": []}
        if len(texts) != len(ids):
            raise ValueError("ids and texts must have equal length")
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError("texts and metadatas must have equal length")
        item_ids = _mutation_items(list(ids), "update_metadata")

        packed = [pack_metadata(t, m) for t, m in zip(texts, metadatas)]
        return self._update_items(
            [{"item_id": i, "metadata": m} for i, m in zip(item_ids, packed)],
            partition_name=partition_name,
            await_completion=await_completion,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            **kwargs,
        )

    def update_documents(
        self,
        ids: List[Any],
        documents: List[Document],
        *,
        vectors: Optional[List[List[float]]] = None,
        update_vectors: bool = True,
        partition_name: Optional[str] = None,
        await_completion: Optional[bool] = None,
        timeout_s: Optional[float] = None,
        poll_interval_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Replace existing items' vector and stored content from Documents.

        pyenvector >= 1.6 can replace an item's vector in place, so the new
        `page_content` is re-embedded by default and both the vector and the
        packed payload are swapped, preserving the item IDs. Pass
        ``update_vectors=False`` for a metadata-only update, or supply
        pre-computed `vectors` when this store has no embeddings configured.
        """
        texts = [getattr(d, "page_content", "") for d in documents]
        metadatas = [getattr(d, "metadata", {}) for d in documents]

        if not update_vectors:
            return self.update_metadata(
                ids,
                texts,
                metadatas,
                partition_name=partition_name,
                await_completion=await_completion,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
                **kwargs,
            )

        if not ids:
            return {"request_id": [], "not_found_item_ids": []}
        if len(texts) != len(ids):
            raise ValueError("ids and documents must have equal length")
        if vectors is None:
            if self._embeddings is None:
                raise ValueError(
                    "embeddings is None and vectors not provided; pass `vectors` "
                    "or use update_vectors=False for a metadata-only update"
                )
            vectors = self._embeddings.embed_documents(texts)
        if len(vectors) != len(ids):
            raise ValueError("ids and vectors must have equal length")

        item_ids = _mutation_items(list(ids), "update_documents")
        packed = [pack_metadata(t, m) for t, m in zip(texts, metadatas)]
        return self._update_items(
            [
                {"item_id": i, "vector": v, "metadata": m}
                for i, v, m in zip(item_ids, vectors, packed)
            ],
            partition_name=partition_name,
            await_completion=await_completion,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            **kwargs,
        )

    def upsert_documents(
        self,
        documents: List[Document],
        ids: Optional[List[Any]] = None,
        *,
        vectors: Optional[List[List[float]]] = None,
        partition_name: Optional[str] = None,
        await_completion: Optional[bool] = None,
        timeout_s: Optional[float] = None,
        poll_interval_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Insert and update Documents in one call (pyenvector >= 1.6.0).

        `ids` is positional against `documents`: an entry that is `None`
        inserts (enVector issues the item ID), and an entry that carries an
        existing item ID replaces that item in place. Omit `ids` entirely to
        insert everything. A caller-chosen ID cannot create a new item — an ID
        matching no live row comes back in ``not_found_item_ids`` rather than
        being inserted under that ID.

        Returns the merged SDK result: ``{"request_id": [...],
        "inserted_item_ids": [...], "not_found_item_ids": [...]}``, where
        ``inserted_item_ids`` maps positionally onto the ID-less entries.
        """
        if not documents:
            return {
                "request_id": [],
                "inserted_item_ids": [],
                "not_found_item_ids": [],
            }
        if ids is not None and len(ids) != len(documents):
            raise ValueError("ids and documents must have equal length")

        texts = [getattr(d, "page_content", "") for d in documents]
        metadatas = [getattr(d, "metadata", {}) for d in documents]
        if vectors is None:
            if self._embeddings is None:
                raise ValueError("embeddings is None and vectors not provided")
            vectors = self._embeddings.embed_documents(texts)
        if len(vectors) != len(documents):
            raise ValueError("documents and vectors must have equal length")

        packed = [pack_metadata(t, m) for t, m in zip(texts, metadatas)]
        raw_ids: List[Optional[Any]] = (
            list(ids) if ids is not None else [None] * len(documents)
        )
        item_ids = [
            None if x is None else _mutation_items([x], "upsert_documents")[0]
            for x in raw_ids
        ]

        specs = [
            {"item_id": i, "vector": v, "metadata": m}
            for i, v, m in zip(item_ids, vectors, packed)
        ]
        return self._mutate(
            "upsert",
            specs,
            partition_name=partition_name,
            await_completion=await_completion,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            **kwargs,
        )

    def _drain_pending_inserts(
        self,
        index: Any,
        timeout_s: Optional[float] = None,
        poll_interval_s: Optional[float] = None,
    ) -> None:
        """Wait for un-awaited inserts to merge before mutating their rows.

        Updating or upserting a row whose insert has not reached the merged
        stage makes that row disappear from search, even though the call reports
        success and the index still counts it. Measured on a 1.6 stack: 4 of 6
        mixed upserts lost the updated row when the insert had not been waited
        for, 0 of 6 when it had.

        Inserts stay fast because the wait is paid here — once, and only when
        rows are actually mutated — rather than on every insert. Search and
        delete are unaffected and need no wait.
        """
        if not self._pending_inserts:
            return
        w = self.config.write
        pending = self._pending_inserts
        self._pending_inserts = {}
        try:
            for partition_name, request_ids in pending.items():
                if not request_ids:
                    continue
                index.wait_for_insert_stage(
                    request_ids=request_ids,
                    target_stage="segmentation",
                    timeout_s=w.timeout_s if timeout_s is None else timeout_s,
                    poll_interval_s=(
                        w.poll_interval_s
                        if poll_interval_s is None
                        else poll_interval_s
                    ),
                    partition_name=partition_name,
                )
        except Exception:
            # Not drained: put them back so the next mutation tries again rather
            # than silently mutating rows that are still unmerged.
            for partition_name, request_ids in pending.items():
                self._pending_inserts.setdefault(partition_name, []).extend(request_ids)
            raise

    def _update_items(
        self, specs: List[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        return self._mutate("update", specs, **kwargs)

    def _mutate(
        self,
        op: str,
        specs: List[Dict[str, Any]],
        *,
        partition_name: Optional[str] = None,
        await_completion: Optional[bool] = None,
        timeout_s: Optional[float] = None,
        poll_interval_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run Index.update/upsert over `specs`, chunked to the SDK per-call cap.

        `specs` are plain dicts so the SDK's UpdateItem/UpsertItem dataclasses
        are resolved lazily, keeping the module usable without pyenvector.

        Chunks are separate server transactions: if a later chunk fails, the
        earlier ones stay applied.
        """
        update_item, upsert_item = _try_import_item_types()
        item_cls = update_item if op == "update" else upsert_item
        index = self._loaded_index()
        w = self.config.write
        self._drain_pending_inserts(index, timeout_s, poll_interval_s)

        merged: Dict[str, Any] = {
            "request_id": [],
            "inserted_item_ids": [],
            "not_found_item_ids": [],
        }
        for chunk in _chunked(specs, MAX_MUTATION_ITEMS_PER_CALL):
            result = getattr(index, op)(
                [item_cls(**spec) for spec in chunk],
                await_completion=(
                    w.await_update if await_completion is None else await_completion
                ),
                timeout_s=w.timeout_s if timeout_s is None else timeout_s,
                poll_interval_s=(
                    w.poll_interval_s if poll_interval_s is None else poll_interval_s
                ),
                partition_name=partition_name,
                **kwargs,
            )
            result = result or {}
            if result.get("request_id"):
                merged["request_id"].append(result["request_id"])
            merged["inserted_item_ids"].extend(result.get("inserted_item_ids") or [])
            merged["not_found_item_ids"].extend(result.get("not_found_item_ids") or [])

        if op == "update":
            merged.pop("inserted_item_ids")
        return merged

    # -------------------------------
    # Partitions (pyenvector >= 1.5.0)
    # -------------------------------
    def create_partition(self, partition_name: str) -> Any:
        """Create a named partition in the index."""
        return self.client.index.create_partition(partition_name)

    def drop_partition(self, partition_name: str) -> Any:
        """Drop a named partition from the index (its data is removed)."""
        return self.client.index.drop_partition(partition_name)

    def list_partitions(self) -> Any:
        """List the index's partitions as dicts {name, status, num_vectors}."""
        return self.client.index.list_partitions()

    def _similarity_search_with_scores(
        self,
        *,
        embedding: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        top_k = fetch_k or self.config.index.fetch_k or k

        try:
            results = self._loaded_index().search(
                query=embedding,
                top_k=top_k,
                output_fields=self.config.index.output_fields,
                partition_names=partition_names,
            )
        except Exception as e:  # narrow-matched below, re-raised otherwise
            # Deleting every row races the backend's shard bookkeeping: a search
            # in that window answers NotFound instead of the empty result a
            # never-populated index returns. Normalise the two — but only after
            # the server confirms the index really holds nothing, so a transient
            # NotFound over live data surfaces as the error it is instead of
            # being silently reported as "no matches".
            if not (_is_empty_shard_list_error(e) and self._index_is_empty()):
                raise
            return []
        # pyenvector Index.search returns a list for each query; we passed single query
        result = (
            results[0]
            if isinstance(results, list) and results and isinstance(results[0], list)
            else results
        )

        if not result:
            return []

        docs_with_scores: List[Tuple[Document, float]] = []
        # Iterate from top-1 to top-k
        for item in result:
            # item = {"id": ..., "score": float, "metadata": [str] or {...}}
            score = float(item.get("score", 0.0))
            md_obj_raw = item.get("metadata")
            if md_obj_raw in (None, "", [], {}):
                # Skip placeholder/empty hits returned by the backend.
                continue

            # Metadata encryption/decryption is handled by the SDK.
            # Envector currently supports a single associated data field (string).
            # Convention: if the string is JSON like {"text": str, "metadata": {...}},
            # we unpack it; otherwise, we treat the raw string as the document text.
            md_obj = unpack_metadata(md_obj_raw)

            text = md_obj.get("text", "") if "_raw" not in md_obj else md_obj["_raw"]
            metadata = md_obj.get("metadata", {}) if "_raw" not in md_obj else {}
            if not text and not metadata:
                # Treat empty text+metadata as no result.
                continue

            # client-side filter
            if filter:
                # simple dict-equality filter on top-level user metadata
                matched = all(metadata.get(k) == v for k, v in filter.items())
                if not matched:
                    continue
            if score_threshold is not None and score < score_threshold:
                continue

            doc_id = item.get("id")
            doc = Document(
                page_content=text,
                metadata={**metadata, "_score": score, "_id": item.get("id")},
                id=doc_id,
            )
            docs_with_scores.append((doc, score))

        # Trim to k after filtering
        return docs_with_scores[:k]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search similar items for a text query.

        - Embeds query if embeddings are provided; else expect `embedding` kwarg.
        - Applies optional client-side filter and score threshold.
        """
        embedding: Optional[List[float]] = kwargs.pop("embedding", None)
        if embedding is None:
            if self._embeddings is None:
                raise ValueError("embeddings is None and no `embedding` provided")
            embedding = self._embeddings.embed_query(query)

        docs_with_scores = self._similarity_search_with_scores(
            embedding=embedding,
            k=k,
            filter=filter,
            score_threshold=score_threshold,
            fetch_k=fetch_k,
            partition_names=partition_names,
            **kwargs,
        )
        return [
            Document(
                page_content=doc.page_content,
                metadata={
                    k: v for k, v in doc.metadata.items() if k not in ("_score", "_id")
                },
                id=getattr(doc, "id", None),
            )
            for doc, _ in docs_with_scores
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        embedding: Optional[List[float]] = kwargs.pop("embedding", None)
        if embedding is None:
            if self._embeddings is None:
                raise ValueError("embeddings is None and no `embedding` provided")
            embedding = self._embeddings.embed_query(query)

        return self._similarity_search_with_scores(
            embedding=embedding,
            k=k,
            filter=filter,
            score_threshold=score_threshold,
            fetch_k=fetch_k,
            partition_names=partition_names,
            **kwargs,
        )

    # Vector-based variant required by some VectorStore interfaces
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        *,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_with_scores = self._similarity_search_with_scores(
            embedding=embedding,
            k=k,
            filter=filter,
            score_threshold=score_threshold,
            fetch_k=fetch_k,
            partition_names=partition_names,
            **kwargs,
        )
        return [doc for doc, _ in docs_with_scores]

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        *,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        return self._similarity_search_with_scores(
            embedding=embedding,
            k=k,
            filter=filter,
            score_threshold=score_threshold,
            fetch_k=fetch_k,
            partition_names=partition_names,
            **kwargs,
        )

    # -------------------------------
    # Class constructors (LangChain compatibility)
    # -------------------------------
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        *,
        vectors: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> List[int]:
        """Insert a list of Documents.

        Mirrors LangChain's VectorStore API. Delegates to `add_texts` by
        extracting `page_content` and `metadata` from each Document.

        Notes:
        - Manual `ids` are ignored: enVector issues its own item IDs, and a
          caller-chosen ID cannot create a new item. To overwrite existing
          items, pass their returned IDs to `update_documents` or
          `upsert_documents`.
        - When `embeddings` is not configured, you must supply `vectors`.
        - The returned item IDs are durable and addressable: use them with
          `delete`, `update_documents` and `upsert_documents`.
        """
        texts = [getattr(d, "page_content", "") for d in documents]
        metadatas = [getattr(d, "metadata", {}) for d in documents]
        return self.add_texts(
            texts=texts, metadatas=metadatas, ids=ids, vectors=vectors, **kwargs
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        *,
        embeddings: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "Envector":  # type: ignore[override]
        """Create a store from texts. Requires `config` in kwargs.

        Remaining keyword arguments are forwarded to `add_texts`, so a store
        with no embeddings can be seeded with pre-computed `vectors`.

        Example:
            Envector.from_texts(texts, metadatas=..., embeddings=..., config=cfg)
        """
        config: Optional[EnvectorConfig] = kwargs.pop("config", None)  # type: ignore
        client: Optional[EnvectorClient] = kwargs.pop("client", None)  # type: ignore
        if config is None:
            raise ValueError("`config` (EnvectorConfig) is required for from_texts().")
        store = cls(config=config, embeddings=embeddings, client=client)
        # Everything left over belongs to add_texts: `vectors` for a store with
        # no embeddings, plus partition_name and the write-path overrides.
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        *,
        embeddings: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> "Envector":  # type: ignore[override]
        texts = [d.page_content for d in documents]
        metadatas = [getattr(d, "metadata", {}) for d in documents]
        return cls.from_texts(
            texts=texts, metadatas=metadatas, embeddings=embeddings, **kwargs
        )

    # Optional: if LangChain is installed, this will be used; otherwise, users may call similarity_search directly.
    def as_retriever(self, **kwargs: Any):  # pragma: no cover - wrapper
        try:
            from langchain_core.vectorstores import VectorStoreRetriever  # type: ignore

            return VectorStoreRetriever(vectorstore=self, **kwargs)
        except Exception:
            # Minimal shim if VectorStoreRetriever is unavailable
            class _Retriever:
                def __init__(
                    self, vs: Envector, search_kwargs: Optional[Dict[str, Any]] = None
                ):
                    self.vs = vs
                    self.search_kwargs = search_kwargs or {}

                def get_relevant_documents(self, query: str) -> List[Document]:
                    return self.vs.similarity_search(query, **self.search_kwargs)

            return _Retriever(self, kwargs.get("search_kwargs"))
