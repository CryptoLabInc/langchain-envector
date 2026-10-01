"""Integration coverage for `get_by_ids` against a live server.

The LangChain standard tests cover the plain read-back. This adds what they do
not: reading right after an un-awaited insert, a delete becoming invisible as
soon as it returns, named partitions, and an index with metadata encryption.

Run with:
    ENVECTOR_ADDRESS=host:port ENVECTOR_KEY_PATH=./keys ENVECTOR_KEY_ID=my_key \\
        pytest -q -m integration tests/integration_tests/test_get_by_ids.py
"""

from __future__ import annotations

import os
import secrets
from typing import Generator, List

import pytest

from langchain_envector.config import (
    ConnectionConfig,
    EnvectorConfig,
    IndexSettings,
    KeyConfig,
    WriteSettings,
)
from langchain_envector.vectorstore import SDK_HAS_GET_BY_IDS, Document, Envector

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not SDK_HAS_GET_BY_IDS, reason="installed pyenvector has no Index.get_by_ids"
    ),
]

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


@pytest.fixture(params=[False, True], ids=["plain-metadata", "encrypted-metadata"])
def store(request) -> Generator[Envector, None, None]:
    name = f"lc_gbi_{secrets.token_hex(4)}"
    cfg = EnvectorConfig(
        connection=ConnectionConfig(address=_require_env("ENVECTOR_ADDRESS")),
        key=KeyConfig(
            key_path=_require_env("ENVECTOR_KEY_PATH"),
            key_id=_require_env("ENVECTOR_KEY_ID"),
        ),
        index=IndexSettings(
            index_name=name, dim=DIM, metadata_encryption=request.param
        ),
        # Nothing here waits for a merge: get_by_ids must not need one.
        write=WriteSettings(await_insert=False, await_delete=False),
        create_if_missing=True,
    )
    store = Envector(config=cfg)
    try:
        yield store
    finally:
        try:
            store.client.ev.delete_index(name)
        except Exception:
            pass


def _add(store: Envector, texts: List[str], **kwargs) -> List[str]:
    return store.add_texts(
        texts,
        metadatas=[{"n": i} for i in range(len(texts))],
        vectors=[_unit_vector(i) for i in range(len(texts))],
        **kwargs,
    )


def test_readable_right_after_insert_and_gone_right_after_delete(
    store: Envector,
) -> None:
    ids = _add(store, ["a", "b", "c"])

    docs = store.get_by_ids(ids)
    assert docs == [
        Document(page_content=t, metadata={"n": i}, id=ids[i])
        for i, t in enumerate("abc")
    ]

    # Missing and foreign ids are left out; order follows the request.
    got = store.get_by_ids([ids[2], "999999", "not-an-id", ids[0]])
    assert [d.id for d in got] == [ids[2], ids[0]]

    store.delete([ids[1]])
    assert [d.id for d in store.get_by_ids(ids)] == [ids[0], ids[2]]


def test_named_partition_needs_partition_name(store: Envector) -> None:
    default_ids = _add(store, ["default doc"])
    store.create_partition("tenant_a")
    tenant_ids = _add(store, ["tenant doc"], partition_name="tenant_a")

    in_tenant = store.get_by_ids(tenant_ids, partition_name="tenant_a")
    assert [d.page_content for d in in_tenant] == ["tenant doc"]

    # Item ids restart per partition: the same number in the default partition
    # is a different document, which is why partition_name matters.
    assert tenant_ids == default_ids
    assert [d.page_content for d in store.get_by_ids(tenant_ids)] == ["default doc"]
