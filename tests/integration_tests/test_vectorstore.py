from __future__ import annotations

import math
import os
import secrets
from typing import Generator

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_envector.config import (
    ConnectionConfig,
    EnvectorConfig,
    IndexSettings,
    KeyConfig,
)
from langchain_envector.vectorstore import Envector

pytestmark = pytest.mark.integration


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"Set {name} to enable integration test")
    return value


class _UnitNorm(Embeddings):
    """Scale another embedding's vectors to unit length."""

    def __init__(self, inner: Embeddings) -> None:
        self._inner = inner

    @staticmethod
    def _unit(v: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._unit(v) for v in self._inner.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._unit(self._inner.embed_query(text))


class TestEnvectorVectorStore(VectorStoreIntegrationTests):
    # VectorStoreIntegrationTests provides the standard search/add/get scenarios;
    # this class only wires up the Envector fixture and capability flags.
    @staticmethod
    def get_embeddings() -> Embeddings:
        # Envector requires dimension in [32, 4096]. Scores are inner products
        # computed under encryption, which assumes unit-norm vectors; the fake
        # embedding's raw Gaussian components (up to ~3) rank documents wrongly.
        return _UnitNorm(DeterministicFakeEmbedding(size=32))

    @property
    def has_async(self) -> bool:
        # Envector does not yet support async methods.
        return False

    @property
    def has_get_by_ids(self) -> bool:
        # Envector does not yet support get by IDs.
        return False

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore[override]
        # Set up Envector vector store for testing.
        address = _require_env("ENVECTOR_ADDRESS")
        key_path = _require_env("ENVECTOR_KEY_PATH")
        key_id = _require_env("ENVECTOR_KEY_ID")
        index_name = f"lc_std_{secrets.token_hex(4)}"

        cfg = EnvectorConfig(
            connection=ConnectionConfig(address=address),
            key=KeyConfig(key_path=key_path, key_id=key_id),
            index=IndexSettings(
                index_name=index_name, dim=32, query_encryption="plain"
            ),
            create_if_missing=True,
        )
        # Create the vector store.
        store = Envector(config=cfg, embeddings=self.get_embeddings())

        try:
            yield store
        finally:
            try:
                # Clean up: delete the created index.
                store.client.ev.delete_index(index_name)
            except Exception:
                pass

    # Capability gaps are marked xfail, but each override still runs the real
    # standard test through super(), so an XPASS means the gap actually closed.
    # These four need arbitrary caller-chosen IDs to survive add_documents;
    # enVector issues its own item IDs and can only update existing ones.
    @pytest.mark.xfail(
        reason="arbitrary caller-chosen IDs cannot be created: enVector issues "
        "its own item IDs, so add_documents(ids=...) only updates existing items."
    )
    def test_deleting_documents(self, vectorstore: VectorStore) -> None:
        super().test_deleting_documents(vectorstore)

    @pytest.mark.xfail(
        reason="arbitrary caller-chosen IDs cannot be created: enVector issues "
        "its own item IDs, so add_documents(ids=...) only updates existing items."
    )
    def test_deleting_bulk_documents(self, vectorstore: VectorStore) -> None:
        super().test_deleting_bulk_documents(vectorstore)

    @pytest.mark.xfail(
        reason="arbitrary caller-chosen IDs cannot be created; overwriting works "
        "only with item IDs the store issued."
    )
    def test_add_documents_by_id_with_mutation(self, vectorstore: VectorStore) -> None:
        super().test_add_documents_by_id_with_mutation(vectorstore)

    @pytest.mark.xfail(
        reason="arbitrary caller-chosen IDs cannot be created, so adding twice "
        "under the same caller IDs inserts twice."
    )
    def test_add_documents_with_ids_is_idempotent(
        self, vectorstore: VectorStore
    ) -> None:
        super().test_add_documents_with_ids_is_idempotent(vectorstore)

    # Async standard tests are not overridden: has_async=False makes the base
    # class skip them.
