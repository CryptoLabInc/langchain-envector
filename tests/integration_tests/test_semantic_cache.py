"""Integration coverage for `EnvectorSemanticCache` against a live server.

The unit tests pin the cache's logic on fakes; this checks what only the
server can answer: a search scoped to a partition that exists but holds no
rows, a row being found right after `update` returns, scores under CKKS
coming out on the right side of the threshold, partitions keeping models
apart, and `clear` dropping partitions. Runs once with plain and once with
encrypted metadata.

Run with:
    ENVECTOR_ADDRESS=host:port ENVECTOR_KEY_PATH=./keys ENVECTOR_KEY_ID=my_key \\
        pytest -q -m integration tests/integration_tests/test_semantic_cache.py

The key bundle must carry a ``MetadataKey.json`` for the encrypted-metadata
run; pyenvector's auto key setup writes one only when generating with
``metadata_encryption=True``.
"""

from __future__ import annotations

import math
import os
import secrets
from typing import Dict, Generator, List

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.load import dumps
from langchain_core.outputs import ChatGeneration, Generation

from langchain_envector.cache import PARTITION_PREFIX, EnvectorSemanticCache
from langchain_envector.config import (
    ConnectionConfig,
    EnvectorConfig,
    IndexSettings,
    KeyConfig,
    WriteSettings,
)

pytestmark = pytest.mark.integration

DIM = 32
THRESHOLD = 0.9

# Prompts with known cosine similarity to FRANCE (the first coordinate).
FRANCE = "What is the capital of France?"
FRANCE_AGAIN = "Tell me the capital city of France."  # 0.95: hit
GERMANY = "What is the capital of Germany?"  # 0.6: miss
NEAR_MISS = "Which city is France's capital?"  # 0.85: miss at 0.9


def _vec(cos: float, axis: int = 1) -> List[float]:
    v = [0.0] * DIM
    v[0] = cos
    v[axis] = math.sqrt(1.0 - cos * cos)
    return v


PROMPT_VECTORS: Dict[str, List[float]] = {
    FRANCE: _vec(1.0),
    FRANCE_AGAIN: _vec(0.95),
    GERMANY: _vec(0.6),
    NEAR_MISS: _vec(0.85),
}

LLM_A = '{"model": "a"}---[("stop", None)]'
LLM_B = '{"model": "b"}---[("stop", None)]'
ANSWER = [Generation(text="Paris")]


class PromptEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(PROMPT_VECTORS[t]) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return list(PROMPT_VECTORS[text])


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"Set {name} to enable integration test")
    return value


@pytest.fixture(params=[False, True], ids=["plain-metadata", "encrypted-metadata"])
def cache(request) -> Generator[EnvectorSemanticCache, None, None]:
    name = f"lc_cache_{secrets.token_hex(4)}"
    cfg = EnvectorConfig(
        connection=ConnectionConfig(address=_require_env("ENVECTOR_ADDRESS")),
        key=KeyConfig(
            key_path=_require_env("ENVECTOR_KEY_PATH"),
            key_id=_require_env("ENVECTOR_KEY_ID"),
        ),
        index=IndexSettings(
            index_name=name, dim=DIM, metadata_encryption=request.param
        ),
        # A cache must not need to wait for a merge: `update` returns and
        # the next `lookup` has to see the row.
        write=WriteSettings(await_insert=False),
        create_if_missing=True,
    )
    cache = EnvectorSemanticCache(
        config=cfg, embeddings=PromptEmbeddings(), similarity_threshold=THRESHOLD
    )
    try:
        yield cache
    finally:
        try:
            cache.vectorstore.client.ev.delete_index(name)
        except Exception:
            pass


def _partitions(cache: EnvectorSemanticCache) -> List[str]:
    return sorted(
        p["name"]
        for p in cache.vectorstore.list_partitions()
        if str(p["name"]).startswith(PARTITION_PREFIX)
    )


def test_miss_then_hit_right_after_update(cache: EnvectorSemanticCache) -> None:
    # No partition yet: a miss without touching the server's search path.
    assert cache.lookup(FRANCE, LLM_A) is None
    assert _partitions(cache) == []

    cache.update(FRANCE, LLM_A, ANSWER)
    assert cache.lookup(FRANCE, LLM_A) == ANSWER
    assert cache.lookup(FRANCE_AGAIN, LLM_A) == ANSWER
    assert cache.lookup(NEAR_MISS, LLM_A) is None
    assert cache.lookup(GERMANY, LLM_A) is None


def test_lookup_in_a_partition_with_no_rows_is_a_miss(
    cache: EnvectorSemanticCache,
) -> None:
    # The partition exists (as after a crash between create and insert, or
    # one made by another process) but holds nothing. The server may answer
    # the search with an error instead of an empty result; either is a miss.
    cache.update(GERMANY, LLM_B, [Generation(text="Berlin")])  # index not empty
    from langchain_envector.cache import _partition_name

    cache.vectorstore.create_partition(_partition_name(LLM_A))
    assert cache.lookup(FRANCE, LLM_A) is None


def test_chat_generations_round_trip(cache: EnvectorSemanticCache) -> None:
    prompt = dumps([SystemMessage(content="Be terse."), HumanMessage(content=FRANCE)])
    PROMPT_VECTORS[f"system: Be terse.\nhuman: {FRANCE}"] = _vec(1.0, axis=2)
    answer = [ChatGeneration(message=AIMessage(content="Paris."))]
    cache.update(prompt, LLM_A, answer)
    got = cache.lookup(prompt, LLM_A)
    assert got == answer
    assert isinstance(got[0], ChatGeneration)


def test_llm_strings_are_kept_apart(cache: EnvectorSemanticCache) -> None:
    cache.update(FRANCE, LLM_A, ANSWER)
    assert cache.lookup(FRANCE, LLM_B) is None
    other = [Generation(text="Paris, from B")]
    cache.update(FRANCE, LLM_B, other)
    assert cache.lookup(FRANCE, LLM_A) == ANSWER
    assert cache.lookup(FRANCE, LLM_B) == other
    assert len(_partitions(cache)) == 2


def test_clear_one_llm_string_then_everything(cache: EnvectorSemanticCache) -> None:
    cache.update(FRANCE, LLM_A, ANSWER)
    cache.update(FRANCE, LLM_B, ANSWER)
    cache.vectorstore.create_partition("tenant_a")

    cache.clear(llm_string=LLM_A)
    assert cache.lookup(FRANCE, LLM_A) is None
    assert cache.lookup(FRANCE, LLM_B) == ANSWER
    assert len(_partitions(cache)) == 1

    cache.clear()
    assert _partitions(cache) == []
    assert cache.lookup(FRANCE, LLM_B) is None
    names = [p["name"] for p in cache.vectorstore.list_partitions()]
    assert "tenant_a" in names

    # The cache is usable again after a clear.
    cache.update(FRANCE, LLM_A, ANSWER)
    assert cache.lookup(FRANCE, LLM_A) == ANSWER


def test_two_caches_on_one_index_survive_each_others_clear(
    cache: EnvectorSemanticCache,
) -> None:
    # A second instance on the same index, as two processes would have.
    other = EnvectorSemanticCache(
        config=cache.vectorstore.config,
        embeddings=PromptEmbeddings(),
        similarity_threshold=THRESHOLD,
    )
    cache.update(FRANCE, LLM_A, ANSWER)
    assert other.lookup(FRANCE, LLM_A) == ANSWER

    other.clear()
    # The first instance still remembers the partition name; the server does not.
    assert cache.lookup(FRANCE, LLM_A) is None
    cache.update(FRANCE, LLM_A, ANSWER)
    assert cache.lookup(FRANCE, LLM_A) == ANSWER
    assert other.lookup(FRANCE, LLM_A) == ANSWER

    other.clear(llm_string=LLM_A)
    cache.update(GERMANY, LLM_A, [Generation(text="Berlin")])
    assert cache.lookup(GERMANY, LLM_A) == [Generation(text="Berlin")]
    assert cache.lookup(FRANCE, LLM_A) is None


def test_clear_after_another_instance_cleared_is_a_no_op(
    cache: EnvectorSemanticCache,
) -> None:
    other = EnvectorSemanticCache(
        config=cache.vectorstore.config,
        embeddings=PromptEmbeddings(),
        similarity_threshold=THRESHOLD,
    )
    cache.update(FRANCE, LLM_A, ANSWER)
    cache.update(FRANCE, LLM_B, ANSWER)
    other.clear()
    # This instance still remembers both partitions; the server has neither.
    cache.clear(llm_string=LLM_A)
    cache.clear()
    assert _partitions(cache) == []
    cache.update(FRANCE, LLM_A, ANSWER)
    assert cache.lookup(FRANCE, LLM_A) == ANSWER
