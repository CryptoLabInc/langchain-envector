from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pyenvector import UpdateItem, UpsertItem
from pyenvector.index.index import MAX_MUTATION_ITEMS_PER_CALL

from .config import EnvectorConfig
from .client import EnvectorClient
from .types import Embeddings, as_embeddings, pack_metadata, unpack_metadata


def _mutation_items(
    item_ids: List[Any], label: str, *, dedupe: bool = False
) -> List[int]:
    """Coerce caller-supplied IDs to the ``int`` item_ids the SDK addresses.

    The SDK rejects non-positive and repeated ids with its own message; both
    are caught here and named after the calling method. ``dedupe=True``
    (delete) drops repeats instead, since deleting a row twice is deleting it.
    """
    try:
        ints = [int(x) for x in item_ids]
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Envector.{label} expects integer item IDs (or numeric strings) "
            "as returned by add_texts/add_documents."
        ) from e
    if any(i <= 0 for i in ints):
        raise ValueError(
            f"Envector.{label}: item IDs are positive integers (got {min(ints)})."
        )
    if dedupe:
        return list(dict.fromkeys(ints))
    if len(set(ints)) != len(ints):
        raise ValueError(f"Envector.{label}: item IDs must be unique within one call.")
    return ints


def _split_caller_ids(ids: List[Any]) -> Tuple[List[Optional[int]], List[Any]]:
    """Sort caller-supplied IDs into enVector item IDs and everything else.

    Returns ``(item_ids, foreign)``: ``item_ids`` is positional against ``ids``
    with ``None`` wherever the entry was ``None`` or not an integer, and
    ``foreign`` lists the non-integer values so the caller can be told they
    were not honoured.
    """
    item_ids: List[Optional[int]] = []
    foreign: List[Any] = []
    for x in ids:
        if x is None:
            item_ids.append(None)
            continue
        try:
            value = int(x)
        except (TypeError, ValueError):
            value = 0
        if value <= 0:  # the server issues positive ints only
            item_ids.append(None)
            foreign.append(x)
        else:
            item_ids.append(value)
    return item_ids, foreign


def _one_embedding_arg(embedding: Any, embeddings: Any) -> Any:
    """Resolve the standard positional ``embedding`` and our older
    ``embeddings=`` keyword into one value, rejecting conflicting pairs."""
    if (
        isinstance(embedding, (list, tuple))
        and embedding
        and isinstance(embedding[0], dict)
    ):
        # The second positional argument used to be `metadatas`. Catch the old
        # call shape here rather than failing deeper inside as_embeddings().
        raise TypeError(
            "the second positional argument is `embedding` (as in LangChain's "
            "VectorStore); pass metadatas by keyword: metadatas=[...]"
        )
    if embedding is not None and embeddings is not None and embedding is not embeddings:
        raise ValueError("pass either `embedding` or `embeddings`, not both")
    return embedding if embedding is not None else embeddings


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


class Envector(VectorStore):
    """LangChain-compatible VectorStore adaptor for Envector.

    This class wraps the high-level `pyenvector` SDK (`EnvectorClient`, `Index`).
    The one place the package reads below that surface is `client.py`, which
    reuses the SDK's process-wide connection across stores.
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

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """The embedding model used for queries, or ``None`` when the store
        works with pre-computed vectors only."""
        return self._embeddings

    def _index_is_empty(self) -> bool:
        """Ask the server whether this index currently holds no rows.

        Used to decide whether an "index has no shards" search error means the
        index is genuinely empty. Any failure to answer counts as "not known to
        be empty", so the original error is re-raised rather than swallowed.
        """
        try:
            summary = self.client.index.summary()
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
    ) -> List[str]:
        """Add texts to the index and return their item IDs.

        Texts are embedded with the configured model, or pass pre-computed
        ``vectors``. ``partition_name`` targets a named partition. Inserted rows
        are searchable when the call returns; ``await_completion=True`` also
        waits for the server to merge and save them (default
        ``config.write.await_insert``). Other keyword arguments go to
        ``Index.insert`` and apply to the rows this call inserts; the upsert
        arm below takes only ``timeout_s`` / ``poll_interval_s``.

        ``ids`` follows LangChain's add-or-update contract as far as enVector
        allows: an entry that is an item ID (int or numeric str, such as the
        ``Document.id`` search results carry) updates that item in place; an ID
        with no live row, or a non-integer ID, cannot be created, so that row is
        inserted with a server-issued ID and a ``UserWarning``. ``None`` entries
        insert. The returned list holds the IDs actually in the index, as
        strings like LangChain's ``Document.id``; every method here accepts
        them back in that form.
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

        # The SDK waits up to a day for an insert; WriteSettings.timeout_s is
        # sized for delete/update and is not applied here unless passed in.
        timeout_s = kwargs.pop("timeout_s", None)
        poll_interval_s = kwargs.pop("poll_interval_s", None)

        if ids is not None:
            if len(ids) != len(texts):
                raise ValueError("texts and ids must have equal length")
            item_ids, foreign = _split_caller_ids(list(ids))
            if foreign:
                warnings.warn(
                    f"Envector cannot insert under caller-chosen IDs; {len(foreign)} "
                    f"of the given ids are not enVector item IDs and were ignored "
                    f"(e.g. {foreign[0]!r}). Those rows were inserted with "
                    "server-issued IDs — use the returned IDs to address them.",
                    UserWarning,
                    stacklevel=2,
                )
            if any(i is not None for i in item_ids):
                return self._add_or_update(
                    texts,
                    metadatas,
                    item_ids,
                    vectors=vectors,
                    partition_name=partition_name,
                    await_completion=await_completion,
                    timeout_s=timeout_s,
                    poll_interval_s=poll_interval_s,
                    insert_kwargs=kwargs,
                )

        # Prepare metadata JSON strings per item
        packed = [pack_metadata(t, m) for t, m in zip(texts, metadatas)]

        w = self.config.write
        awaited = w.await_insert if await_completion is None else await_completion
        request_ids: List[str] = kwargs.pop("request_ids", [])
        waits: Dict[str, Any] = {}
        if timeout_s is not None:
            waits["timeout_s"] = timeout_s
        if poll_interval_s is not None:
            waits["poll_interval_s"] = poll_interval_s
        item_ids = self.client.index.insert(
            data=vectors,
            metadata=packed,
            partition_name=partition_name,
            request_ids=request_ids,
            await_completion=awaited,
            **waits,
            **kwargs,
        )
        # Only a merge can be waited for later, and ``execute_until="flush"``
        # stops before one is submitted.
        merging = kwargs.get("execute_until", "segmentation") == "segmentation"
        if not awaited and request_ids and merging:
            self._pending_inserts.setdefault(partition_name, []).extend(request_ids)
        return [str(i) for i in item_ids]

    def _add_or_update(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        item_ids: List[Optional[int]],
        *,
        vectors: List[List[float]],
        partition_name: Optional[str],
        await_completion: Optional[bool],
        timeout_s: Optional[float],
        poll_interval_s: Optional[float],
        insert_kwargs: Dict[str, Any],
    ) -> List[str]:
        """`add_texts` when some entries name existing item IDs.

        Routes everything through one ``upsert_documents`` call — ``None``
        slots insert, ID slots update in place — then re-inserts any document
        whose ID matched no live row, since the server cannot recreate an item
        under a chosen ID. Returns the item IDs really in the index, positional
        against ``texts``.
        """
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        result = self.upsert_documents(
            docs,
            ids=item_ids,
            vectors=vectors,
            partition_name=partition_name,
            await_completion=await_completion,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
        inserted = iter(result.get("inserted_item_ids") or [])
        out: List[str] = [str(next(inserted) if i is None else i) for i in item_ids]

        missing = set(result.get("not_found_item_ids") or [])
        if missing:
            pos = [k for k, i in enumerate(item_ids) if i in missing]
            warnings.warn(
                f"{len(pos)} of the given item IDs match no live row "
                f"(e.g. {item_ids[pos[0]]}); enVector cannot recreate an item under "
                "a chosen ID, so those documents were inserted as new rows. The "
                "returned list holds the IDs actually used.",
                UserWarning,
                stacklevel=3,
            )
            new_ids = self.add_texts(
                [texts[k] for k in pos],
                [metadatas[k] for k in pos],
                vectors=[vectors[k] for k in pos],
                partition_name=partition_name,
                await_completion=await_completion,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
                **insert_kwargs,
            )
            for k, nid in zip(pos, new_ids):
                out[k] = nid
        return out

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

        Unlike update/upsert this does not wait for pending inserts first: the
        server's DeleteDataCore classifies a not-yet-merged row as Pending and
        applies the delete when the row lands (late-binding through
        InsertShardMapList), so deleting an unmerged row is safe.
        """
        if not ids:
            return False
        item_ids = _mutation_items(ids, "delete", dedupe=True)

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
        present = iter(
            _mutation_items([x for x in raw_ids if x is not None], "upsert_documents")
        )
        item_ids = [None if x is None else next(present) for x in raw_ids]

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
        self, index: Any, partition_name: Optional[str] = None
    ) -> None:
        """Wait for a partition's un-awaited inserts to merge before mutating its rows.

        Updating a row whose insert has not merged yet drops it from search
        while the call still reports success, so the wait is paid here — once,
        and only when rows are mutated — rather than on every insert. Updates
        and deletes only reach rows of the partition they name (the default
        one when none is named), so only that partition's inserts are waited
        for. The wait uses ``config.write.drain_timeout_s`` because it grows
        with the number of pending batches. Timing out raises and keeps the
        pending ids, so nothing is mutated until a retry drains them.
        """
        request_ids = self._pending_inserts.pop(partition_name, None)
        if not request_ids:
            return
        w = self.config.write
        try:
            index.wait_for_insert_stage(
                request_ids=request_ids,
                target_stage="segmentation",
                timeout_s=w.drain_timeout_s,
                poll_interval_s=w.poll_interval_s,
                partition_name=partition_name,
            )
        except Exception:
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
        item_cls = UpdateItem if op == "update" else UpsertItem
        index = self._loaded_index()
        w = self.config.write
        awaited = w.await_update if await_completion is None else await_completion
        self._drain_pending_inserts(index, partition_name)

        merged: Dict[str, Any] = {
            "request_id": [],
            "inserted_item_ids": [],
            "not_found_item_ids": [],
        }
        for chunk in _chunked(specs, MAX_MUTATION_ITEMS_PER_CALL):
            result = getattr(index, op)(
                [item_cls(**spec) for spec in chunk],
                await_completion=awaited,
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
            if op == "upsert" and not awaited and result.get("insert_request_id"):
                # Rows the upsert inserted merge like any other insert; queue
                # them so a following update does not catch them unmerged.
                self._pending_inserts.setdefault(partition_name, []).append(
                    result["insert_request_id"]
                )

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
        search_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        if kwargs:
            # Anything the SDK would not see must not disappear quietly.
            raise TypeError(
                f"unexpected search keyword arguments: {sorted(kwargs)}. Known: "
                "k, filter, score_threshold, fetch_k, partition_names, search_params."
            )
        top_k = fetch_k or self.config.index.fetch_k or k

        try:
            results = self._loaded_index().search(
                query=embedding,
                top_k=top_k,
                output_fields=self.config.index.output_fields,
                search_params=search_params,
                partition_names=partition_names,
            )
        except Exception as e:  # narrow-matched below, re-raised otherwise
            # A search right after every row was deleted can answer NotFound
            # instead of an empty result. Treat it as empty only once the server
            # confirms the index holds nothing; otherwise re-raise.
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
                metadata=metadata,
                id=str(doc_id) if doc_id is not None else None,
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
        search_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search similar items for a text query.

        - Embeds query if embeddings are provided; else expect `embedding` kwarg.
        - Applies optional client-side filter and score threshold.
        - ``search_params`` goes to ``Index.search`` as is (e.g. ``{"nprobe": 64}``
          for IVF indexes). Unknown keyword arguments raise ``TypeError``.
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
            search_params=search_params,
            **kwargs,
        )
        return [doc for doc, _ in docs_with_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        partition_names: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None,
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
            search_params=search_params,
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
        search_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_with_scores = self._similarity_search_with_scores(
            embedding=embedding,
            k=k,
            filter=filter,
            score_threshold=score_threshold,
            fetch_k=fetch_k,
            partition_names=partition_names,
            search_params=search_params,
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
        search_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        return self._similarity_search_with_scores(
            embedding=embedding,
            k=k,
            filter=filter,
            score_threshold=score_threshold,
            fetch_k=fetch_k,
            partition_names=partition_names,
            search_params=search_params,
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
    ) -> List[str]:
        """Add or update a list of Documents.

        Mirrors LangChain's VectorStore API. Delegates to `add_texts` by
        extracting `page_content` and `metadata` from each Document.

        Notes:
        - As in the base class, when `ids` is not given and the Documents carry
          an ``id`` (search results do), those ids are used. Ids that are
          enVector item IDs update the item in place; see `add_texts` for what
          happens to ids enVector cannot honour.
        - When `embeddings` is not configured, you must supply `vectors`.
        - The returned item IDs are durable and addressable: use them with
          `delete`, `update_documents` and `upsert_documents`.
        """
        if ids is None:
            doc_ids = [getattr(d, "id", None) for d in documents]
            if any(doc_ids):
                ids = doc_ids  # type: ignore[assignment]
        texts = [getattr(d, "page_content", "") for d in documents]
        metadatas = [getattr(d, "metadata", {}) for d in documents]
        return self.add_texts(
            texts=texts, metadatas=metadatas, ids=ids, vectors=vectors, **kwargs
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        *,
        embeddings: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "Envector":  # type: ignore[override]
        """Create a store from texts. Requires `config` in kwargs.

        Takes the embedding model as the second positional argument like every
        other LangChain vector store (and like the inherited `afrom_texts`,
        which calls it that way); `embeddings=` is kept as a keyword alias.
        Remaining keyword arguments are forwarded to `add_texts`, so a store
        with no embeddings can be seeded with pre-computed `vectors`.

        Example:
            Envector.from_texts(texts, emb, metadatas=..., config=cfg)
        """
        embedding = _one_embedding_arg(embedding, embeddings)
        config: Optional[EnvectorConfig] = kwargs.pop("config", None)  # type: ignore
        client: Optional[EnvectorClient] = kwargs.pop("client", None)  # type: ignore
        if config is None:
            raise ValueError("`config` (EnvectorConfig) is required for from_texts().")
        store = cls(config=config, embeddings=embedding, client=client)
        # Everything left over belongs to add_texts: `vectors` for a store with
        # no embeddings, plus partition_name and the write-path overrides.
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        *,
        embeddings: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> "Envector":  # type: ignore[override]
        """Create a store from Documents. Same argument shape as `from_texts`."""
        embedding = _one_embedding_arg(embedding, embeddings)
        if "ids" not in kwargs:
            doc_ids = [getattr(d, "id", None) for d in documents]
            if any(doc_ids):
                kwargs["ids"] = doc_ids
        texts = [d.page_content for d in documents]
        metadatas = [getattr(d, "metadata", {}) for d in documents]
        return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)

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
