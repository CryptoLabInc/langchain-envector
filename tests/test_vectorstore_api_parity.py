"""Parity tests for the LangChain ``VectorStore`` methods Envector does not
implement yet.

Written ahead of the implementation. Each test below is marked
``xfail(strict=True)``: it runs on every ``pytest``, fails today for the stated
reason, and the moment an implementation makes it pass ``strict=True`` turns the
XPASS into a failure — so the marker cannot be forgotten. When a feature lands,
remove the marker from its tests, not the tests.

Scope — the gaps that can be closed inside this package and are still open:

1. ``_select_relevance_score_fn`` → ``similarity_search_with_relevance_scores``
   and ``search_type="similarity_score_threshold"``
2. MMR — ``max_marginal_relevance_search`` and its three siblings

(The constructor-signature and ``embeddings``-property gaps have been closed;
their tests now live in ``test_vectorstore.py`` as ordinary regression tests.)

``get_by_ids`` and native async are deliberately absent: the first needs a wire
API that addresses rows by item ID (``GetMetadataRequest`` carries only
shard/row positions) and the second needs an async pyenvector. Neither can be
made to pass from this package.

Facts the tests lean on, verified against pyenvector 1.6.2:

- Search scores are raw inner products (the server RPC is ``InnerProduct``) and
  pyenvector does not normalise vectors. With unit-norm embeddings the score is
  therefore a cosine similarity in [-1, 1]; the fixtures below use unit-norm
  vectors so a relevance function built on that assumption is exercised across
  the whole range, negative included.
- A search hit carries ``id``, ``score`` and ``metadata`` only. The wire
  ``Metadata`` message is ``{id, data}`` — the stored vector never comes back.
  MMR has to get candidate vectors some other way; re-embedding the returned
  texts with the configured embeddings is the path these tests assume, so the
  fake embeddings return the same vector for the same text every time.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from langchain_envector.config import (
    ConnectionConfig,
    EnvectorConfig,
    IndexSettings,
    KeyConfig,
)
from langchain_envector.vectorstore import Document, Envector

from .conftest import FakeClient, FakeIndex


# ---------------------------------------------------------------------------
# Markers — one per gap, each naming the hook that closes it.
# ---------------------------------------------------------------------------

xfail_relevance = pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason="Envector does not define _select_relevance_score_fn, so the "
    "inherited similarity_search_with_relevance_scores raises.",
)

xfail_mmr = pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason="Envector does not implement max_marginal_relevance_search; the "
    "base class raises.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Unit-norm corpus in 4 dimensions. Query is [1, 0, 0, 0], so the inner
# products are exactly the first coordinates: two near-duplicates at the top,
# two orthogonal rows, and one row pointing away from the query.
CORPUS: Dict[str, List[float]] = {
    "apple pie": [1.0, 0.0, 0.0, 0.0],
    "apple tart": [0.995, 0.0998, 0.0, 0.0],
    "bicycle": [0.0, 1.0, 0.0, 0.0],
    "harbour": [0.0, 0.0, 1.0, 0.0],
    "antimatter": [-0.5, 0.0, 0.0, 0.866],
}
QUERY = [1.0, 0.0, 0.0, 0.0]


def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


class LookupEmbeddings:
    """Embeddings that return a fixed vector per text.

    Deterministic across calls, which is what lets MMR re-embed the texts a
    search returned and get back the vectors the index scored.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(CORPUS[t]) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return list(QUERY)


@dataclass
class ScoringFakeIndex(FakeIndex):
    """FakeIndex whose ``search`` actually scores ``CORPUS`` by inner product.

    Returns the top ``top_k`` hits in the shape the real SDK uses — ``id``,
    ``score``, ``metadata`` and nothing else — so ``top_k`` genuinely limits
    what the store gets to see. That is what makes the ``fetch_k`` assertions
    meaningful: MMR at ``k=2`` cannot find a diverse pair unless it asked the
    server for more than 2.
    """

    corpus: Dict[str, List[float]] = field(default_factory=lambda: dict(CORPUS))

    def search(
        self,
        query: List[float],
        top_k: int,
        output_fields: List[str],
        partition_names: Optional[List[str]] = None,
    ):
        self.searched.append({"top_k": top_k, "partition_names": partition_names})
        texts = list(self.corpus)
        ranked = sorted(
            texts, key=lambda t: _dot(query, self.corpus[t]), reverse=True
        )
        hits: List[Dict[str, Any]] = [
            {
                "id": texts.index(t) + 1,
                "score": _dot(query, self.corpus[t]),
                "metadata": json.dumps({"text": t, "metadata": {"src": t}}),
            }
            for t in ranked[:top_k]
        ]
        return [hits]


def _cfg() -> EnvectorConfig:
    return EnvectorConfig(
        connection=ConnectionConfig(address="dummy:0"),
        key=KeyConfig(key_path="./keys", key_id="kid"),
        index=IndexSettings(index_name="idx", dim=4),
    )


def _store(index: Optional[FakeIndex] = None, *, with_embeddings: bool = True):
    index = index or ScoringFakeIndex()
    emb = LookupEmbeddings() if with_embeddings else None
    return Envector(config=_cfg(), embeddings=emb, client=FakeClient(index)), index


def _texts(docs: List[Document]) -> List[str]:
    return [d.page_content for d in docs]


# ---------------------------------------------------------------------------
# 1. Relevance scores
# ---------------------------------------------------------------------------


@xfail_relevance
def test_relevance_scores_stay_in_the_unit_interval_and_keep_rank_order():
    store, _ = _store()

    # The base class warns when a score leaves [0, 1]; treat that as failure.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pairs = store.similarity_search_with_relevance_scores("q", k=5)

    assert _texts([d for d, _ in pairs]) == [
        "apple pie",
        "apple tart",
        "bicycle",
        "harbour",
        "antimatter",
    ]
    scores = [s for _, s in pairs]
    assert all(0.0 <= s <= 1.0 for s in scores), scores
    # Raw inner products 1.0 > 0.995 > 0 == 0 > -0.5 must map monotonically.
    assert scores[0] > scores[1] > scores[2] == scores[3] > scores[4], scores


@xfail_relevance
def test_relevance_score_threshold_cuts_below_the_given_relevance():
    store, _ = _store()

    pairs = store.similarity_search_with_relevance_scores("q", k=5)
    cutoff = pairs[1][1]  # relevance of "apple tart"

    kept = store.similarity_search_with_relevance_scores(
        "q", k=5, score_threshold=cutoff
    )
    assert _texts([d for d, _ in kept]) == ["apple pie", "apple tart"]
    assert all(s >= cutoff for _, s in kept)


@xfail_relevance
def test_search_dispatches_similarity_score_threshold():
    store, _ = _store()

    docs = store.search("q", "similarity_score_threshold", k=5, score_threshold=0.9)

    assert docs, "expected at least the exact match to clear a 0.9 threshold"
    assert docs[0].page_content == "apple pie"
    assert "antimatter" not in _texts(docs)
    assert "_score" not in docs[0].metadata and "_id" not in docs[0].metadata


@xfail_relevance
def test_retriever_with_similarity_score_threshold_returns_documents():
    store, _ = _store()
    retriever = store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.9, "k": 5},
    )

    docs = retriever.invoke("q")

    assert docs and docs[0].page_content == "apple pie"
    assert "antimatter" not in _texts(docs)


@xfail_relevance
async def test_async_relevance_scores_match_sync():
    store, _ = _store()

    sync_pairs = store.similarity_search_with_relevance_scores("q", k=3)
    async_pairs = await store.asimilarity_search_with_relevance_scores("q", k=3)

    assert [(d.page_content, s) for d, s in async_pairs] == [
        (d.page_content, s) for d, s in sync_pairs
    ]


# ---------------------------------------------------------------------------
# 2. MMR
# ---------------------------------------------------------------------------


@xfail_mmr
def test_mmr_returns_k_distinct_documents_without_internal_keys():
    store, _ = _store()

    docs = store.max_marginal_relevance_search("q", k=2, fetch_k=5)

    assert len(docs) == 2
    assert len({d.id for d in docs}) == 2
    for d in docs:
        assert "_score" not in d.metadata and "_id" not in d.metadata


@xfail_mmr
def test_mmr_over_fetches_by_fetch_k():
    # The diversity trade-off needs candidates beyond k; the store must ask the
    # server for fetch_k rows, not k.
    store, index = _store()

    store.max_marginal_relevance_search("q", k=2, fetch_k=5, lambda_mult=0.5)

    assert index.searched[-1]["top_k"] == 5


@xfail_mmr
def test_mmr_with_lambda_one_is_plain_similarity_order():
    store, _ = _store()

    docs = store.max_marginal_relevance_search("q", k=2, fetch_k=5, lambda_mult=1.0)

    assert _texts(docs) == ["apple pie", "apple tart"]


@xfail_mmr
def test_mmr_with_lambda_zero_avoids_the_near_duplicate():
    # Pure diversity: after the top hit, the next pick is whatever is least
    # like it. "apple tart" is nearly identical to "apple pie" and must lose;
    # "antimatter" points away from it and must win.
    store, _ = _store()

    docs = store.max_marginal_relevance_search("q", k=2, fetch_k=5, lambda_mult=0.0)

    assert docs[0].page_content == "apple pie"
    assert docs[1].page_content == "antimatter"


@xfail_mmr
def test_mmr_by_vector_takes_the_query_embedding_directly():
    store, _ = _store()

    docs = store.max_marginal_relevance_search_by_vector(
        QUERY, k=2, fetch_k=5, lambda_mult=0.0
    )

    assert _texts(docs) == ["apple pie", "antimatter"]


@xfail_mmr
def test_mmr_without_embeddings_fails_with_a_clear_message():
    # A store built for pre-computed vectors has nothing to re-embed candidates
    # with, and the server never returns stored vectors. That must surface as a
    # ValueError naming the missing piece, not as an AttributeError on None.
    store, _ = _store(with_embeddings=False)

    with pytest.raises(ValueError, match="embeddings"):
        store.max_marginal_relevance_search_by_vector(QUERY, k=2, fetch_k=5)


@xfail_mmr
def test_search_dispatches_mmr():
    store, _ = _store()

    docs = store.search("q", "mmr", k=2, fetch_k=5, lambda_mult=0.0)

    assert _texts(docs) == ["apple pie", "antimatter"]


@xfail_mmr
def test_retriever_with_mmr_returns_documents():
    store, _ = _store()
    retriever = store.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 5, "lambda_mult": 0.0}
    )

    docs = retriever.invoke("q")

    assert _texts(docs) == ["apple pie", "antimatter"]


@xfail_mmr
async def test_async_mmr_matches_sync():
    store, _ = _store()

    sync_docs = store.max_marginal_relevance_search("q", k=2, fetch_k=5, lambda_mult=0.0)
    async_docs = await store.amax_marginal_relevance_search(
        "q", k=2, fetch_k=5, lambda_mult=0.0
    )

    assert _texts(async_docs) == _texts(sync_docs)
