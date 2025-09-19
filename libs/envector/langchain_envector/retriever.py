"""Optional custom retriever wrapper.

Prefer using VectorStore.as_retriever, which returns LangChain's VectorStoreRetriever
when available. This module exists as an extension point if custom behavior is needed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .vectorstore import Envector, Document


class EnvectorRetriever:
    def __init__(self, store: Envector, *, search_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.store = store
        self.search_kwargs = search_kwargs or {}

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.store.similarity_search(query, **self.search_kwargs)
