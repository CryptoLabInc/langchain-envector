"""Integration coverage for the pyenvector 1.6 write paths.

`insert` was the only write path before 1.6. This exercises the rest of them
against a live server: in-place `update` (metadata-only and vector+metadata),
mixed `upsert`, and `delete` — each followed by a search that must observe the
change, which is what verifies the await-completion defaults in
`config.write` actually make a write visible to the next read.

Run with:
    ENVECTOR_ADDRESS=host:port ENVECTOR_KEY_PATH=./keys ENVECTOR_KEY_ID=my_key \
        pytest -q -m integration tests/integration_tests/test_write_paths.py
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
)
from langchain_envector.vectorstore import Document, Envector

pytestmark = pytest.mark.integration

DIM = 32


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"Set {name} to enable integration test")
    return value


def _unit_vector(pos: int) -> List[float]:
    """A one-hot vector, so each item is its own unambiguous nearest neighbour."""
    vec = [0.0] * DIM
    vec[pos % DIM] = 1.0
    return vec


@pytest.fixture()
def store() -> Generator[Envector, None, None]:
    address = _require_env("ENVECTOR_ADDRESS")
    key_path = _require_env("ENVECTOR_KEY_PATH")
    key_id = _require_env("ENVECTOR_KEY_ID")
    index_name = f"lc_write_{secrets.token_hex(4)}"

    cfg = EnvectorConfig(
        connection=ConnectionConfig(address=address),
        key=KeyConfig(key_path=key_path, key_id=key_id),
        index=IndexSettings(index_name=index_name, dim=DIM, query_encryption="plain"),
        create_if_missing=True,
    )
    # No embeddings: every vector is supplied explicitly so the assertions below
    # depend only on enVector's behaviour, not on an embedding model.
    vs = Envector(config=cfg, embeddings=None)
    try:
        yield vs
    finally:
        try:
            vs.client.ev.drop_index(index_name)
        except Exception:
            pass


def _top1(store: Envector, vector: List[float]):
    hits = store.similarity_search_with_score_by_vector(vector, k=1)
    return hits[0] if hits else (None, None)


def test_insert_is_searchable_without_sleeping(store: Envector) -> None:
    # Index.insert publishes the rows through its own load step, so this holds
    # with WriteSettings.await_insert off — no sleep and no merge wait.
    vectors = [_unit_vector(i) for i in range(3)]
    ids = store.add_texts(
        ["alpha", "beta", "gamma"],
        metadatas=[{"n": 0}, {"n": 1}, {"n": 2}],
        vectors=vectors,
    )
    assert len(ids) == 3
    assert all(isinstance(i, int) for i in ids)

    # No wait, no sleep: the search must already see all three.
    for pos, (item_id, text) in enumerate(zip(ids, ["alpha", "beta", "gamma"])):
        doc, score = _top1(store, vectors[pos])
        assert doc is not None, f"item {item_id} not searchable right after insert"
        assert doc.page_content == text
        assert score > 0.99


def test_update_metadata_keeps_the_vector(store: Envector) -> None:
    vectors = [_unit_vector(0), _unit_vector(1)]
    ids = store.add_texts(
        ["draft", "other"], metadatas=[{"status": "draft"}, {}], vectors=vectors
    )

    result = store.update_metadata([ids[0]], ["final"], metadatas=[{"status": "final"}])
    assert result["not_found_item_ids"] == []

    # Same vector, new payload: the item still matches its original embedding.
    doc, score = _top1(store, vectors[0])
    assert score > 0.99
    assert doc.page_content == "final"
    assert doc.metadata["status"] == "final"


def test_update_documents_replaces_the_vector(store: Envector) -> None:
    old_vector, other = _unit_vector(0), _unit_vector(1)
    ids = store.add_texts(["old", "other"], vectors=[old_vector, other])

    new_vector = _unit_vector(5)
    result = store.update_documents(
        [ids[0]],
        [Document(page_content="new", metadata={"v": 2})],
        vectors=[new_vector],
    )
    assert result["not_found_item_ids"] == []

    # The new vector resolves to the same item_id...
    doc, score = _top1(store, new_vector)
    assert score > 0.99
    assert doc.page_content == "new"
    assert doc.metadata["_id"] == ids[0]
    # ...and the replaced vector no longer has a near-1 match.
    _, old_score = _top1(store, old_vector)
    assert old_score is None or old_score < 0.99


def test_update_reports_missing_ids_instead_of_raising(store: Envector) -> None:
    ids = store.add_texts(["kept"], vectors=[_unit_vector(0)])
    absent = max(ids) + 999_999

    result = store.update_metadata([absent], ["ignored"])
    assert absent in result["not_found_item_ids"]


def test_upsert_mixes_inserts_and_updates(store: Envector) -> None:
    first, second = _unit_vector(0), _unit_vector(1)
    ids = store.add_texts(["one", "two"], vectors=[first, second])

    fresh = _unit_vector(7)
    result = store.upsert_documents(
        [
            Document(page_content="one-updated", metadata={"r": "update"}),
            Document(page_content="three", metadata={"r": "insert"}),
        ],
        ids=[ids[0], None],
        vectors=[first, fresh],
    )
    assert result["not_found_item_ids"] == []
    assert len(result["inserted_item_ids"]) == 1

    doc, score = _top1(store, first)
    assert score > 0.99
    assert doc.page_content == "one-updated"

    doc, score = _top1(store, fresh)
    assert score > 0.99
    assert doc.page_content == "three"


def test_delete_removes_the_item_from_search(store: Envector) -> None:
    vectors = [_unit_vector(0), _unit_vector(1)]
    ids = store.add_texts(["doomed", "kept"], vectors=vectors)

    assert store.delete([ids[0]]) is True

    # delete awaited the shard rebuild, so the deleted item is already gone.
    _, score = _top1(store, vectors[0])
    assert score is None or score < 0.99

    doc, score = _top1(store, vectors[1])
    assert doc is not None and doc.page_content == "kept"
    assert score > 0.99


def test_delete_accepts_document_ids(store: Envector) -> None:
    vector, survivor = _unit_vector(0), _unit_vector(1)
    store.add_texts(["by-doc-id", "kept"], vectors=[vector, survivor])

    doc, _ = _top1(store, vector)
    # Document.id is the str form of the item_id; delete must accept it as-is.
    assert isinstance(doc.id, str)
    assert store.delete([doc.id]) is True

    _, score = _top1(store, vector)
    assert score is None or score < 0.99


def test_search_after_deleting_every_item_is_empty(store: Envector) -> None:
    # Emptying an index leaves it with no shards, and the backend answers a
    # search over it with NotFound rather than an empty result. The store
    # normalises that back to [], matching a never-populated index.
    vectors = [_unit_vector(0), _unit_vector(1)]
    ids = store.add_texts(["a", "b"], vectors=vectors)

    assert store.delete(ids) is True
    assert store.similarity_search_by_vector(vectors[0], k=2) == []


def test_delete_of_missing_ids_is_a_noop(store: Envector) -> None:
    ids = store.add_texts(["kept"], vectors=[_unit_vector(0)])

    assert store.delete([max(ids) + 999_999]) is True
    assert store.delete([]) is False
    assert store.delete(None) is False


class _FixedEmbeddings:
    """Embeds everything to the same vector: retrieval reduces to "is it wired up"."""

    def embed_documents(self, texts):
        return [_unit_vector(0) for _ in texts]

    def embed_query(self, text):
        return _unit_vector(0)


def test_as_retriever_reaches_the_server(store: Envector) -> None:
    # as_retriever is a README entry point with no coverage. The fixture store
    # has no embeddings (the retriever must embed the query), so build one that
    # does over the same index.
    embedded = Envector(config=store.config, embeddings=_FixedEmbeddings())
    embedded.add_texts(["retrievable"])

    retriever = embedded.as_retriever(search_kwargs={"k": 1})
    docs = retriever.invoke("anything")
    assert [d.page_content for d in docs] == ["retrievable"]


def test_from_texts_creates_and_populates(store: Envector) -> None:
    # from_texts needs its own index, and Envector.from_texts takes the config
    # through kwargs; reuse the fixture's config with a fresh index name.
    cfg = store.config
    index_name = f"{cfg.index.index_name}_ft"
    sub = EnvectorConfig(
        connection=cfg.connection,
        key=cfg.key,
        index=IndexSettings(index_name=index_name, dim=DIM, query_encryption="plain"),
        create_if_missing=True,
    )
    try:
        created = Envector.from_texts(
            ["one", "two"],
            metadatas=[{"n": 1}, {"n": 2}],
            config=sub,
            vectors=[_unit_vector(0), _unit_vector(1)],
        )
        docs = created.similarity_search_by_vector(_unit_vector(1), k=1)
        assert [d.page_content for d in docs] == ["two"]
    finally:
        try:
            store.client.ev.drop_index(index_name)
        except Exception:
            pass
