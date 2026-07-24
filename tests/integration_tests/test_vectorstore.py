from __future__ import annotations

import os
import secrets
from typing import Generator

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
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


class TestEnvectorVectorStore(VectorStoreIntegrationTests):
    # VectorStoreIntegrationTests provides the standard search/add/get scenarios;
    # this class only wires up the Envector fixture and capability flags.
    @staticmethod
    def get_embeddings() -> DeterministicFakeEmbedding:
        # Envector requires dimension in [32, 4096].
        return DeterministicFakeEmbedding(size=32)

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

    # Note: test_delete_missing_content is NOT overridden — Envector.delete
    # (pyenvector >= 1.4) accepts numeric-string IDs and must not raise for
    # missing items, which is exactly what the standard test verifies.

    @pytest.mark.xfail(
        reason="Standard test requires user-provided IDs to be honored by "
        "add_documents; Envector assigns its own item IDs (delete itself is "
        "supported with returned item IDs)."
    )
    def test_deleting_documents(self, vectorstore: VectorStore) -> None:
        pass

    @pytest.mark.xfail(
        reason="Standard test requires user-provided IDs to be honored by "
        "add_documents; Envector assigns its own item IDs (delete itself is "
        "supported with returned item IDs)."
    )
    def test_deleting_bulk_documents(self, vectorstore: VectorStore) -> None:
        pass

    @pytest.mark.xfail(
        reason="Standard test requires upsert via user-provided IDs; Envector "
        "assigns its own item IDs (metadata replacement is supported via "
        "update_metadata with returned item IDs, pyenvector >= 1.5)."
    )
    def test_add_documents_by_id_with_mutation(self, vectorstore: VectorStore) -> None:
        pass

    @pytest.mark.xfail(
        reason="Standard test requires idempotent add via user-provided IDs; "
        "Envector assigns its own item IDs."
    )
    def test_add_documents_with_ids_is_idempotent(
        self, vectorstore: VectorStore
    ) -> None:
        pass

    @pytest.mark.xfail(
        reason="Empty index returns placeholder results in current backend."
    )
    def test_vectorstore_is_empty(self, vectorstore: VectorStore) -> None:
        pass

    @pytest.mark.xfail(
        reason="Empty index returns placeholder results in current backend."
    )
    def test_vectorstore_still_empty(self, vectorstore: VectorStore) -> None:
        pass

    # Async standard tests are not overridden: has_async=False makes the base
    # class skip them.
