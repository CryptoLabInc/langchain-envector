from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence
from uuid import uuid4

from .config import EnvectorConfig
from .client import EnvectorClient
from .types import Embeddings, as_embeddings, pack_metadata, unpack_metadata


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
            def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
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
    """LangChain-compatible VectorStore adaptor for Envector (ES2).

    This class wraps the high-level `es2` SDK. It does not use low-level
    gRPC stubs or `es2.api.Indexer` directly.
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
        **kwargs: Any,
    ) -> List[int]:
        """Add texts to the encrypted index.

        If embeddings are provided, the texts are embedded automatically.
        Otherwise, provide pre-computed `vectors`.
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

        # Insert using high-level ES2 Index
        result_ids = self.client.index.insert(data=vectors, metadata=packed)

        # Return ephemeral placeholders to satisfy VectorStore interface,
        # but they are NOT persisted/addressable.
        return result_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search similar items for a text query.

        - Embeds query if embeddings are provided; else expect `embedding` kwarg.
        - Applies optional client-side filter and score threshold.
        """
        embedding: Optional[List[float]] = kwargs.get("embedding")
        if embedding is None:
            if self._embeddings is None:
                raise ValueError("embeddings is None and no `embedding` provided")
            embedding = self._embeddings.embed_query(query)

        top_k = fetch_k or self.config.index.fetch_k or k

        results = self.client.index.search(query=embedding, top_k=top_k, output_fields=self.config.index.output_fields)
        # ES2 Index.search returns a list for each query; we passed single query
        result = results[0] if isinstance(results, list) and results and isinstance(results[0], list) else results

        docs = []
        # Iterate from top-1 to top-k
        for item in result:
            # item = {"id": ..., "score": float, "metadata": [str] or {...}}
            score = float(item.get("score", 0.0))
            md_obj_raw = item.get("metadata")
        
            # Metadata encryption/decryption is handled by the SDK.
            # Envector currently supports a single associated data field (string).
            # Convention: if the string is JSON like {"text": str, "metadata": {...}},
            # we unpack it; otherwise, we treat the raw string as the document text.
            md_obj = unpack_metadata(md_obj_raw)

            text = md_obj.get("text", "") if "_raw" not in md_obj else md_obj["_raw"]
            metadata = md_obj.get("metadata", {}) if "_raw" not in md_obj else {}

            # client-side filter
            if filter:
                # simple dict-equality filter on top-level user metadata
                matched = all(metadata.get(k) == v for k, v in filter.items())
                if not matched:
                    continue
            if score_threshold is not None and score < score_threshold:
                continue

            doc = Document(page_content=text, metadata={**metadata, "_score": score, "_id": item.get("id")})
            docs.append(doc)

        # Trim to k after filtering
        return docs[:k]

    # Vector-based variant required by some VectorStore interfaces
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        *,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return self.similarity_search(
            query="",  # unused
            k=k,
            filter=filter,
            score_threshold=score_threshold,
            fetch_k=fetch_k,
            embedding=embedding,
            **kwargs,
        )

    # -------------------------------
    # Class constructors (LangChain compatibility)
    # -------------------------------
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

        Example:
            Envector.from_texts(texts, metadatas=..., embeddings=..., config=cfg)
        """
        config: Optional[EnvectorConfig] = kwargs.get("config")  # type: ignore
        client: Optional[EnvectorClient] = kwargs.get("client")  # type: ignore
        if config is None:
            raise ValueError("`config` (EnvectorConfig) is required for from_texts().")
        store = cls(config=config, embeddings=embeddings, client=client)
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
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
        return cls.from_texts(texts=texts, metadatas=metadatas, embeddings=embeddings, **kwargs)

    # Optional: if LangChain is installed, this will be used; otherwise, users may call similarity_search directly.
    def as_retriever(self, **kwargs: Any):  # pragma: no cover - wrapper
        try:
            from langchain_core.vectorstores import VectorStoreRetriever  # type: ignore

            return VectorStoreRetriever(vectorstore=self, **kwargs)
        except Exception:
            # Minimal shim if VectorStoreRetriever is unavailable
            class _Retriever:
                def __init__(self, vs: Envector, search_kwargs: Optional[Dict[str, Any]] = None):
                    self.vs = vs
                    self.search_kwargs = search_kwargs or {}

                def get_relevant_documents(self, query: str) -> List[Document]:
                    return self.vs.similarity_search(query, **self.search_kwargs)

            return _Retriever(self, kwargs.get("search_kwargs"))
