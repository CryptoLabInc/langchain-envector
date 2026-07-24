from __future__ import annotations

from langchain_envector.config import (
    ConnectionConfig,
    EnvectorConfig,
    IndexSettings,
    KeyConfig,
)
from langchain_envector.vectorstore import Envector, Document as LC_Document

from .conftest import FakeClient, FakeEmbeddings, FakeIndex


def _cfg() -> EnvectorConfig:
    return EnvectorConfig(
        connection=ConnectionConfig(address="dummy:0"),
        key=KeyConfig(key_path="./keys", key_id="kid"),
        index=IndexSettings(index_name="idx", dim=4),
    )


def test_add_texts_returns_item_ids():
    # Test that add_texts returns the item IDs assigned by the vector store
    # Note that user-provided IDs are ignored
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    ret_ids = store.add_texts(
        ["t1", "t2"], metadatas=[{"m": 1}, {"m": 2}], ids=["a", "b"]
    )  # input ids ignored

    # Returned IDs
    assert len(ret_ids) == 2
    assert ret_ids == [2, 3]

    # Stored metadata must not contain id
    assert len(client.index.inserted) == 1
    packed = client.index.inserted[0]["metadata"]
    assert len(packed) == 2
    assert '"id"' not in packed[0]


def test_similarity_search_with_filter_and_threshold():
    index = FakeIndex()
    # Two items, different scores and tags
    index.search_payload = [
        [
            {
                "id": "pos-0",
                "score": 0.95,
                "metadata": '{"text": "A", "metadata": {"tag": "keep"}}',
            },
            {
                "id": "pos-1",
                "score": 0.40,
                "metadata": '{"text": "B", "metadata": {"tag": "drop"}}',
            },
        ]
    ]
    client = FakeClient(index)
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    docs = store.similarity_search(
        "q", k=5, filter={"tag": "keep"}, score_threshold=0.5
    )
    assert len(docs) == 1
    assert docs[0].page_content == "A"


def test_similarity_search_handles_string_metadata():
    index = FakeIndex()
    # metadata returned as a single JSON string instead of list
    index.search_payload = [
        [
            {
                "id": "pos-0",
                "score": 0.8,
                "metadata": '{"text": "S", "metadata": {"t": 1}}',
            },
        ]
    ]
    client = FakeClient(index)
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    docs = store.similarity_search("q", k=1)
    assert len(docs) == 1
    assert docs[0].page_content == "S"
    assert docs[0].metadata.get("t") == 1


def test_similarity_search_uses_raw_text_when_not_json():
    index = FakeIndex()
    # metadata is a plain string (not JSON); should be treated as page_content
    index.search_payload = [
        [
            {
                "id": "pos-raw",
                "score": 0.6,
                "metadata": "Plain text content without JSON",
            },
        ]
    ]
    client = FakeClient(index)
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    docs = store.similarity_search("q", k=1)
    assert len(docs) == 1
    assert docs[0].page_content == "Plain text content without JSON"
    # user metadata should be empty dict when not provided


def test_similarity_search_handles_python_literal_metadata():
    index = FakeIndex()
    literal = str({"text": "Literal", "metadata": {"tag": "py"}})
    index.search_payload = [
        [
            {"id": "pos-lit", "score": 0.7, "metadata": literal},
        ]
    ]
    client = FakeClient(index)
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    docs = store.similarity_search("q", k=1)
    assert len(docs) == 1
    assert docs[0].page_content == "Literal"
    assert docs[0].metadata.get("tag") == "py"

    # dict-type metadata is not supported currently; only text-based


def test_similarity_search_by_vector_with_filter_and_threshold():
    index = FakeIndex()
    index.search_payload = [
        [
            {
                "id": "v-0",
                "score": 0.88,
                "metadata": '{"text": "Keep", "metadata": {"k": 1}}',
            },
            {
                "id": "v-1",
                "score": 0.30,
                "metadata": '{"text": "Drop", "metadata": {"k": 2}}',
            },
        ]
    ]
    client = FakeClient(index)
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    # Explicit vector search (bypasses embed_query), with filter/threshold
    docs = store.similarity_search_by_vector(
        [0.0, 0.0, 0.0, 0.0], k=5, filter={"k": 1}, score_threshold=0.5
    )
    assert len(docs) == 1
    assert docs[0].page_content == "Keep"
    assert docs[0].metadata["_score"] >= 0.5


def test_similarity_search_with_score_returns_tuples():
    index = FakeIndex()
    index.search_payload = [
        [
            {
                "id": "s-0",
                "score": 0.77,
                "metadata": '{"text": "Doc0", "metadata": {"tag": "x"}}',
            },
            {
                "id": "s-1",
                "score": 0.25,
                "metadata": '{"text": "Doc1", "metadata": {"tag": "y"}}',
            },
        ]
    ]
    client = FakeClient(index)
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    results = store.similarity_search_with_score("query", k=2)
    assert len(results) == 2
    first_doc, first_score = results[0]
    assert isinstance(first_doc, LC_Document)
    assert first_doc.page_content == "Doc0"
    assert first_doc.metadata["_score"] == first_score


def test_similarity_search_with_score_by_vector_returns_tuples():
    index = FakeIndex()
    index.search_payload = [
        [
            {
                "id": "sv-0",
                "score": 0.66,
                "metadata": '{"text": "VectorDoc", "metadata": {"tag": "keep"}}',
            }
        ]
    ]
    client = FakeClient(index)
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    results = store.similarity_search_with_score_by_vector(
        [0.0, 0.0, 0.0, 0.0], k=1, filter={"tag": "keep"}, score_threshold=0.5
    )
    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "VectorDoc"
    assert score == doc.metadata["_score"]


def test_from_texts_inserts_using_embeddings():
    client = FakeClient()
    store = Envector.from_texts(
        ["A", "B"],
        metadatas=[{"m": "a"}, {"m": "b"}],
        embeddings=FakeEmbeddings(dim=4),
        config=_cfg(),
        client=client,
    )
    assert isinstance(store, Envector)
    # One batch inserted
    assert len(client.index.inserted) == 1
    # Two items packed
    assert len(client.index.inserted[0]["metadata"]) == 2


def test_from_documents_paths_through_to_texts():
    client = FakeClient()
    docs = [
        LC_Document(page_content="X", metadata={"a": 1}),
        LC_Document(page_content="Y", metadata={"a": 2}),
    ]
    store = Envector.from_documents(
        docs, embeddings=FakeEmbeddings(dim=4), config=_cfg(), client=client
    )
    assert isinstance(store, Envector)
    assert len(client.index.inserted) == 1
    packed = client.index.inserted[0]["metadata"]
    # Texts preserved
    assert any('"text": "X"' in m for m in packed)
    assert any('"text": "Y"' in m for m in packed)


def test_add_documents_with_embeddings():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    docs = [
        LC_Document(page_content="C1", metadata={"s": 1}),
        LC_Document(page_content="C2", metadata={"s": 2}),
    ]
    ret = store.add_documents(docs)
    assert len(ret) == 2
    assert len(client.index.inserted) == 1
    packed = client.index.inserted[0]["metadata"]
    assert any('"text": "C1"' in m for m in packed)
    assert any('"text": "C2"' in m for m in packed)


def test_add_documents_returns_item_ids():
    # Test that add_documents returns the item IDs assigned by the vector store
    # Note that user-provided IDs are ignored
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    docs = [
        LC_Document(page_content="D1", metadata={"t": 1}),
        LC_Document(page_content="D2", metadata={"t": 2}),
    ]
    ret_ids = store.add_documents(docs, ids=["user-1", "user-2"])

    assert len(ret_ids) == 2
    assert ret_ids == [2, 3]


def test_add_documents_requires_vectors_when_no_embeddings():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=None, client=client)
    docs = [LC_Document(page_content="C", metadata={})]
    try:
        store.add_documents(docs)
        assert (
            False
        ), "Expected ValueError when embeddings is None and no vectors provided"
    except ValueError as e:
        assert "embeddings is None and vectors not provided" in str(e)


def test_delete_passes_item_ids_to_sdk():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)
    ids = store.add_texts(["t1", "t2", "t3"])
    assert ids == [2, 3, 4]

    assert store.delete(ids=[ids[0], ids[2]]) is True
    assert len(client.index.deleted) == 1
    call = client.index.deleted[0]
    assert call["item_ids"] == [2, 4]
    assert call["await_completion"] is False


def test_delete_accepts_string_ids():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)
    ids = store.add_texts(["a", "b"])

    assert store.delete(ids=[str(ids[0])]) is True
    assert client.index.deleted[0]["item_ids"] == [ids[0]]


def test_delete_empty_or_none_returns_false():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    assert store.delete(ids=None) is False
    assert store.delete(ids=[]) is False
    assert client.index.deleted == []


def test_delete_rejects_non_numeric_ids():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)
    try:
        store.delete(ids=["abc"])
        assert False, "Expected ValueError for non-numeric ids"
    except ValueError as e:
        assert "integer item IDs" in str(e)


def test_delete_forwards_await_kwargs():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)
    ids = store.add_texts(["x"])

    store.delete(ids=ids, await_completion=False, timeout_s=12.0, poll_interval_s=0.5)
    call = client.index.deleted[0]
    assert call["await_completion"] is False
    assert call["timeout_s"] == 12.0
    assert call["poll_interval_s"] == 0.5


def test_add_documents_with_explicit_vectors():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=None, client=client)

    docs = [
        LC_Document(page_content="V1", metadata={"k": "a"}),
        LC_Document(page_content="V2", metadata={"k": "b"}),
    ]
    vecs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    ret = store.add_documents(docs, vectors=vecs)
    assert len(ret) == 2
    assert len(client.index.inserted) == 1


def test_add_texts_forwards_partition_name():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    store.add_texts(["t1"], partition_name="tenant_a")
    assert client.index.inserted[0]["partition_name"] == "tenant_a"

    store.add_texts(["t2"])
    assert client.index.inserted[1]["partition_name"] is None


def test_similarity_search_forwards_partition_names():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    store.similarity_search("q", k=1, partition_names=["p1", "p2"])
    assert client.index.searched[0]["partition_names"] == ["p1", "p2"]

    store.similarity_search_with_score("q", k=1)
    assert client.index.searched[1]["partition_names"] is None


def test_delete_forwards_partition_name():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)
    ids = store.add_texts(["x"])

    store.delete(ids=ids, partition_name="tenant_a")
    assert client.index.deleted[0]["partition_name"] == "tenant_a"


def test_update_metadata_packs_and_forwards():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)
    ids = store.add_texts(["old"], metadatas=[{"v": 1}])

    result = store.update_metadata(ids, ["new"], metadatas=[{"v": 2}])
    assert result == {"updated": ids, "skipped": []}

    call = client.index.metadata_updates[0]
    assert call["item_ids"] == ids
    assert '"new"' in call["metadata"][0]
    assert '"v": 2' in call["metadata"][0]
    assert call["partition_name"] is None


def test_update_metadata_validates_lengths_and_ids():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    assert store.update_metadata([], []) == {"updated": [], "skipped": []}

    try:
        store.update_metadata([1, 2], ["only-one"])
        assert False, "Expected ValueError for length mismatch"
    except ValueError as e:
        assert "equal length" in str(e)

    try:
        store.update_metadata(["abc"], ["t"])
        assert False, "Expected ValueError for non-numeric ids"
    except ValueError as e:
        assert "integer item IDs" in str(e)

    assert client.index.metadata_updates == []


def test_update_documents_delegates():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)
    ids = store.add_texts(["old"])

    docs = [LC_Document(page_content="fresh", metadata={"k": "v"})]
    result = store.update_documents(ids, docs, partition_name="p1")
    assert result["updated"] == ids

    call = client.index.metadata_updates[0]
    assert '"fresh"' in call["metadata"][0]
    assert call["partition_name"] == "p1"


def test_partition_management_helpers():
    client = FakeClient()
    store = Envector(config=_cfg(), embeddings=FakeEmbeddings(dim=4), client=client)

    store.create_partition("p1")
    store.create_partition("p2")
    names = [p["name"] for p in store.list_partitions()]
    assert names == ["p1", "p2"]

    store.drop_partition("p1")
    names = [p["name"] for p in store.list_partitions()]
    assert names == ["p2"]
