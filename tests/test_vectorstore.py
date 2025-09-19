from __future__ import annotations

import re

from langchain_envector.config import ConnectionConfig, EnvectorConfig, IndexSettings, KeyConfig
from langchain_envector.vectorstore import Envector

from .conftest import FakeClient, FakeEmbeddings, FakeIndex


def _cfg() -> EnvectorConfig:
    return EnvectorConfig(
        connection=ConnectionConfig(address="dummy:0"),
        key=KeyConfig(key_path="./keys", key_id="kid"),
        index=IndexSettings(index_name="idx", dim=4),
    )


def test_add_texts_ignores_ids_and_returns_ephemeral():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    ret_ids = store.add_texts(["t1", "t2"], metadatas=[{"m": 1}, {"m": 2}], ids=["a", "b"])  # ids ignored

    # Returned IDs are ephemeral placeholders
    assert len(ret_ids) == 2
    assert all(re.match(r"^ephemeral-", i) for i in ret_ids)

    # Stored metadata must not contain id
    assert len(client.index.inserted) == 1
    packed = client.index.inserted[0]["metadata"]
    assert len(packed) == 2
    assert "\"id\"" not in packed[0]


def test_similarity_search_with_filter_and_threshold():
    index = FakeIndex()
    # Two items, different scores and tags
    index.search_payload = [[
        {"id": "pos-0", "score": 0.95, "metadata": "{\"text\": \"A\", \"metadata\": {\"tag\": \"keep\"}}"},
        {"id": "pos-1", "score": 0.40, "metadata": "{\"text\": \"B\", \"metadata\": {\"tag\": \"drop\"}}"},
    ]]
    client = FakeClient(index)
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    docs = store.similarity_search("q", k=5, filter={"tag": "keep"}, score_threshold=0.5)
    assert len(docs) == 1
    assert docs[0].page_content == "A"
    assert docs[0].metadata["_score"] >= 0.5


def test_similarity_search_handles_string_metadata():
    index = FakeIndex()
    # metadata returned as a single JSON string instead of list
    index.search_payload = [[
        {"id": "pos-0", "score": 0.8, "metadata": "{\"text\": \"S\", \"metadata\": {\"t\": 1}}"},
    ]]
    client = FakeClient(index)
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    docs = store.similarity_search("q", k=1)
    assert len(docs) == 1
    assert docs[0].page_content == "S"
    assert docs[0].metadata.get("t") == 1


def test_similarity_search_uses_raw_text_when_not_json():
    index = FakeIndex()
    # metadata is a plain string (not JSON); should be treated as page_content
    index.search_payload = [[
        {"id": "pos-raw", "score": 0.6, "metadata": "Plain text content without JSON"},
    ]]
    client = FakeClient(index)
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    docs = store.similarity_search("q", k=1)
    assert len(docs) == 1
    assert docs[0].page_content == "Plain text content without JSON"
    # user metadata should be empty dict when not provided
    assert all(k in docs[0].metadata for k in ["_score", "_id"])  # only system fields present


def test_similarity_search_handles_python_literal_metadata():
    index = FakeIndex()
    literal = str({"text": "Literal", "metadata": {"tag": "py"}})
    index.search_payload = [[
        {"id": "pos-lit", "score": 0.7, "metadata": literal},
    ]]
    client = FakeClient(index)
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    docs = store.similarity_search("q", k=1)
    assert len(docs) == 1
    assert docs[0].page_content == "Literal"
    assert docs[0].metadata.get("tag") == "py"


    # dict-type metadata is not supported currently; only text-based
