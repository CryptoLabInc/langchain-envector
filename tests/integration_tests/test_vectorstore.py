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

    # test_vectorstore_is_empty and test_vectorstore_still_empty used to be
    # xfailed here for "empty index returns placeholder results". They pass, and
    # the backend's id==0 placeholder filtering is present in 1.5 too, so those
    # xfails were simply stale — nobody noticed because the overrides had `pass`
    # bodies and so reported XPASS without asserting anything.
    #
    # Capability gaps below are marked xfail, but each override still runs the
    # real standard test through super(). A no-op `pass` body would report XPASS
    # without having asserted anything, which reads as "supported now" when
    # nothing was checked.
    #
    # test_delete_missing_content is NOT overridden — Envector.delete accepts
    # numeric-string IDs and must not raise for missing items, which is exactly
    # what the standard test verifies.

    # enVector issues its own item_ids and cannot insert at a caller-chosen ID:
    # Index.upsert routes an id-bearing item to the update arm, and an id
    # matching no live row lands in not_found_item_ids instead of being
    # inserted. delete/update themselves work — see test_e2e.py — but these
    # standard tests require add_documents(ids=...) to be honored.
    @pytest.mark.xfail(
        reason="add_documents(ids=...) is not honored: enVector issues its own "
        "item IDs and cannot insert at a caller-chosen ID."
    )
    def test_deleting_documents(self, vectorstore: VectorStore) -> None:
        super().test_deleting_documents(vectorstore)

    @pytest.mark.xfail(
        reason="add_documents(ids=...) is not honored: enVector issues its own "
        "item IDs and cannot insert at a caller-chosen ID."
    )
    def test_deleting_bulk_documents(self, vectorstore: VectorStore) -> None:
        super().test_deleting_bulk_documents(vectorstore)

    @pytest.mark.xfail(
        reason="add_documents(ids=...) is not honored. Overwriting by ID is "
        "supported through update_documents/upsert_documents with the item IDs "
        "add_documents returned."
    )
    def test_add_documents_by_id_with_mutation(self, vectorstore: VectorStore) -> None:
        super().test_add_documents_by_id_with_mutation(vectorstore)

    @pytest.mark.xfail(
        reason="add_documents(ids=...) is not honored, so adding twice under the "
        "same caller IDs duplicates instead of being idempotent."
    )
    def test_add_documents_with_ids_is_idempotent(
        self, vectorstore: VectorStore
    ) -> None:
        super().test_add_documents_with_ids_is_idempotent(vectorstore)

    # Async standard tests are not overridden: has_async=False makes the base
    # class skip them.
