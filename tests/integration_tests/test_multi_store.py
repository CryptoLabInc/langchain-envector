"""Several Envector stores must be able to coexist in one process.

pyenvector keeps one connection per process: `EnvectorClient.init_connect`
delegates to the `Index.init_connect` classmethod, which disconnects and
replaces the process-global indexer. Without the reuse in
`langchain_envector.client`, building a second store closes the channel the
first one is still holding and every call on it fails with
"Cannot invoke RPC on closed channel!".
"""

from __future__ import annotations

import os
import secrets
from typing import List

import pytest

from langchain_envector.config import (
    ConnectionConfig,
    EnvectorConfig,
    IndexSettings,
    KeyConfig,
)
from langchain_envector.vectorstore import Envector

pytestmark = pytest.mark.integration

DIM = 32


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"Set {name} to enable integration test")
    return value


def _unit_vector(pos: int) -> List[float]:
    vec = [0.0] * DIM
    vec[pos % DIM] = 1.0
    return vec


def _config(index_name: str) -> EnvectorConfig:
    return EnvectorConfig(
        connection=ConnectionConfig(address=_require_env("ENVECTOR_ADDRESS")),
        key=KeyConfig(
            key_path=_require_env("ENVECTOR_KEY_PATH"),
            key_id=_require_env("ENVECTOR_KEY_ID"),
        ),
        index=IndexSettings(index_name=index_name, dim=DIM, query_encryption="plain"),
        create_if_missing=True,
    )


def test_two_stores_stay_usable() -> None:
    name_a = f"lc_multi_a_{secrets.token_hex(4)}"
    name_b = f"lc_multi_b_{secrets.token_hex(4)}"
    store_a = Envector(config=_config(name_a), embeddings=None)
    try:
        store_a.add_texts(["a-data"], vectors=[_unit_vector(0)])

        # Building the second store must not close the first store's channel.
        store_b = Envector(config=_config(name_b), embeddings=None)
        store_b.add_texts(["b-data"], vectors=[_unit_vector(1)])

        # Both stores answer, and each sees only its own index.
        docs_a = store_a.similarity_search_by_vector(_unit_vector(0), k=2)
        docs_b = store_b.similarity_search_by_vector(_unit_vector(1), k=2)
        assert [d.page_content for d in docs_a] == ["a-data"]
        assert [d.page_content for d in docs_b] == ["b-data"]

        # ...and the first store can still write and delete afterwards.
        ids = store_a.add_texts(["a-second"], vectors=[_unit_vector(2)])
        assert store_a.delete(ids) is True
    finally:
        for name in (name_a, name_b):
            try:
                store_a.client.ev.drop_index(name)
            except Exception:
                pass


def test_reopening_the_same_index_sees_persisted_data() -> None:
    name = f"lc_reopen_{secrets.token_hex(4)}"
    first = Envector(config=_config(name), embeddings=None)
    try:
        first.add_texts(["persisted"], vectors=[_unit_vector(0)])

        # A fresh store over an index that already exists: the app-restart path.
        second = Envector(config=_config(name), embeddings=None)
        docs = second.similarity_search_by_vector(_unit_vector(0), k=1)
        assert [d.page_content for d in docs] == ["persisted"]

        ids = second.add_texts(["appended"], vectors=[_unit_vector(1)])
        docs = second.similarity_search_by_vector(_unit_vector(1), k=1)
        assert [d.page_content for d in docs] == ["appended"]
        assert second.delete(ids) is True
    finally:
        try:
            first.client.ev.drop_index(name)
        except Exception:
            pass
