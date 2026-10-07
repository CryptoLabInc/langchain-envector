"""Unit tests for EnvectorSemanticCache against the in-memory fakes."""

from __future__ import annotations

import json
import math
from typing import Dict, List

import pytest
from langchain_core.load import dumps
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, Generation

from langchain_envector.cache import (
    PARTITION_PREFIX,
    EnvectorSemanticCache,
    _partition_name,
    embedding_text,
)
from langchain_envector.config import (
    ConnectionConfig,
    EnvectorConfig,
    IndexSettings,
    KeyConfig,
)

from .conftest import FakeClient, StoringFakeIndex

# Unit-norm prompt vectors in 4 dimensions. Cosine similarity to FRANCE is
# the first coordinate, so each prompt's score against it is known exactly.
FRANCE = "What is the capital of France?"
FRANCE_AGAIN = "Tell me the capital city of France."  # cos 0.95
GERMANY = "What is the capital of Germany?"  # cos 0.6
ON_THRESHOLD = "on the threshold"  # cos 0.9 exactly
BELOW_THRESHOLD = "just below the threshold"  # cos 0.9 - 1e-9

THRESHOLD = 0.9
_BELOW = THRESHOLD - 1e-9

PROMPT_VECTORS: Dict[str, List[float]] = {
    FRANCE: [1.0, 0.0, 0.0, 0.0],
    FRANCE_AGAIN: [0.95, math.sqrt(1 - 0.95**2), 0.0, 0.0],
    GERMANY: [0.6, 0.8, 0.0, 0.0],
    ON_THRESHOLD: [THRESHOLD, math.sqrt(1 - THRESHOLD**2), 0.0, 0.0],
    BELOW_THRESHOLD: [_BELOW, math.sqrt(1 - _BELOW**2), 0.0, 0.0],
}

LLM_A = '{"model": "a"}---[("stop", None)]'
LLM_B = '{"model": "b"}---[("stop", None)]'

ANSWER = [Generation(text="Paris")]
CHAT_ANSWER = [ChatGeneration(message=AIMessage(content="Paris"))]


class PromptEmbeddings:
    """Embeddings that return a fixed vector per prompt and record the calls."""

    def __init__(self, vectors: Dict[str, List[float]] = PROMPT_VECTORS):
        self.vectors = vectors
        self.queries: List[str] = []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise AssertionError("the cache must embed prompts with embed_query")

    def embed_query(self, text: str) -> List[float]:
        self.queries.append(text)
        return list(self.vectors[text])


def _cfg() -> EnvectorConfig:
    return EnvectorConfig(
        connection=ConnectionConfig(address="dummy:0"),
        key=KeyConfig(key_path="./keys", key_id="kid"),
        index=IndexSettings(index_name="idx", dim=4),
    )


def _cache(threshold: float = THRESHOLD, **kwargs) -> EnvectorSemanticCache:
    index = kwargs.pop("index", None) or StoringFakeIndex()
    embeddings = kwargs.pop("embeddings", None) or PromptEmbeddings()
    return EnvectorSemanticCache(
        config=_cfg(),
        embeddings=embeddings,
        similarity_threshold=threshold,
        client=FakeClient(index),
        **kwargs,
    )


def _index(cache: EnvectorSemanticCache) -> StoringFakeIndex:
    return cache.vectorstore.client.index


# -------------------------------
# lookup / update
# -------------------------------


def test_lookup_on_fresh_cache_misses_without_searching_or_creating():
    cache = _cache()
    assert cache.lookup(FRANCE, LLM_A) is None
    assert _index(cache).searched == []
    assert _index(cache).partitions == []


def test_update_then_lookup_round_trips_generations():
    cache = _cache()
    cache.update(FRANCE, LLM_A, ANSWER)
    assert cache.lookup(FRANCE, LLM_A) == ANSWER


def test_chat_generations_round_trip():
    cache = _cache()
    cache.update(FRANCE, LLM_A, CHAT_ANSWER)
    got = cache.lookup(FRANCE, LLM_A)
    assert got == CHAT_ANSWER
    assert isinstance(got[0], ChatGeneration)
    assert got[0].message == AIMessage(content="Paris")


def test_stored_row_holds_prompt_llm_string_and_serialized_generations():
    cache = _cache()
    cache.update(FRANCE, LLM_A, ANSWER)
    (call,) = _index(cache).inserted
    assert call["partition_name"] == _partition_name(LLM_A)
    assert call["partition_name"].startswith(PARTITION_PREFIX)
    assert call["data"] == [PROMPT_VECTORS[FRANCE]]
    (payload,) = call["metadata"]
    stored = json.loads(payload)
    assert stored["text"] == FRANCE
    assert stored["metadata"] == {"llm_string": LLM_A, "return_val": dumps(ANSWER)}


def test_similar_prompt_hits_and_different_prompt_misses():
    cache = _cache()
    cache.update(FRANCE, LLM_A, ANSWER)
    assert cache.lookup(FRANCE_AGAIN, LLM_A) == ANSWER
    assert cache.lookup(GERMANY, LLM_A) is None


@pytest.mark.parametrize(
    "prompt, expected",
    [(ON_THRESHOLD, "hit"), (BELOW_THRESHOLD, "miss")],
    ids=["score == threshold", "score just below threshold"],
)
def test_threshold_boundary_is_inclusive(prompt, expected):
    cache = _cache(THRESHOLD)
    cache.update(FRANCE, LLM_A, ANSWER)
    got = cache.lookup(prompt, LLM_A)
    assert (got == ANSWER) if expected == "hit" else (got is None)


def test_lookup_searches_only_the_partition_for_llm_string():
    cache = _cache()
    cache.update(FRANCE, LLM_A, ANSWER)
    cache.lookup(FRANCE, LLM_A)
    (search,) = _index(cache).searched
    assert search["partition_names"] == [_partition_name(LLM_A)]
    assert search["top_k"] == 4


def test_llm_strings_do_not_see_each_other():
    cache = _cache()
    cache.update(FRANCE, LLM_A, ANSWER)
    assert cache.lookup(FRANCE, LLM_B) is None
    other = [Generation(text="Paris, says B")]
    cache.update(FRANCE, LLM_B, other)
    assert cache.lookup(FRANCE, LLM_A) == ANSWER
    assert cache.lookup(FRANCE, LLM_B) == other
    assert _partition_name(LLM_A) != _partition_name(LLM_B)


def test_partition_is_created_once_per_llm_string():
    cache = _cache()
    cache.update(FRANCE, LLM_A, ANSWER)
    cache.update(GERMANY, LLM_A, [Generation(text="Berlin")])
    cache.update(FRANCE, LLM_B, ANSWER)
    assert sorted(_index(cache).partitions) == sorted(
        [_partition_name(LLM_A), _partition_name(LLM_B)]
    )


def test_partition_that_already_exists_on_the_server_is_reused():
    index = StoringFakeIndex()
    index.partitions.append(_partition_name(LLM_A))
    created = []
    index.create_partition = lambda name: created.append(name)  # type: ignore
    cache = _cache(index=index)
    cache.update(FRANCE, LLM_A, ANSWER)
    assert created == []


def test_create_partition_losing_a_race_is_fine_once_the_partition_exists():
    index = StoringFakeIndex()
    name = _partition_name(LLM_A)

    def racing_create(partition_name):
        # Another process created it between our list and our create.
        index.partitions.append(partition_name)
        raise RuntimeError("partition already exists")

    index.create_partition = racing_create  # type: ignore
    cache = _cache(index=index)
    cache.update(FRANCE, LLM_A, ANSWER)
    assert index.partitions == [name]
    assert cache.lookup(FRANCE, LLM_A) == ANSWER


def test_create_partition_failure_is_raised_when_the_partition_is_missing():
    index = StoringFakeIndex()

    def failing_create(partition_name):
        raise RuntimeError("server said no")

    index.create_partition = failing_create  # type: ignore
    cache = _cache(index=index)
    with pytest.raises(RuntimeError, match="server said no"):
        cache.update(FRANCE, LLM_A, ANSWER)


def test_row_stored_under_another_llm_string_is_not_returned():
    # A hash collision would put another model's row in this partition; the
    # stored llm_string is checked so it is a miss, not a wrong answer.
    cache = _cache()
    cache.update(FRANCE, LLM_A, ANSWER)
    index = _index(cache)
    (key,) = list(index.stored)
    row = json.loads(index.stored[key])
    row["metadata"]["llm_string"] = LLM_B
    index.stored[key] = json.dumps(row)
    assert cache.lookup(FRANCE, LLM_A) is None


def test_update_appends_rather_than_replacing():
    cache = _cache()
    cache.update(FRANCE, LLM_A, ANSWER)
    cache.update(FRANCE, LLM_A, [Generation(text="Paris again")])
    assert len(_index(cache).inserted) == 2
    assert _index(cache).updates == [] and _index(cache).upserts == []


def test_unreadable_cached_generations_warn_and_miss():
    cache = _cache()
    cache.update(FRANCE, LLM_A, ANSWER)
    index = _index(cache)
    (key,) = list(index.stored)
    row = json.loads(index.stored[key])
    row["metadata"]["return_val"] = "not json at all"
    index.stored[key] = json.dumps(row)
    with pytest.warns(UserWarning, match="cannot be deserialized"):
        assert cache.lookup(FRANCE, LLM_A) is None


def test_empty_partition_search_error_is_a_miss_only_when_it_is_empty():
    cache = _cache()
    cache.update(FRANCE, LLM_A, ANSWER)
    index = _index(cache)

    def no_shards(*args, **kwargs):
        raise RuntimeError("shard list for index idx is empty")

    index.search = no_shards  # type: ignore
    # The partition holds a row, so the server's answer is a real failure.
    with pytest.raises(RuntimeError, match="shard list"):
        cache.lookup(FRANCE, LLM_A)
    # Make the partition empty but keep it: now the error means "no rows".
    index.vectors.clear()
    assert cache.lookup(FRANCE, LLM_A) is None


# -------------------------------
# clear
# -------------------------------


def test_clear_llm_string_drops_only_that_partition():
    cache = _cache()
    cache.update(FRANCE, LLM_A, ANSWER)
    cache.update(FRANCE, LLM_B, ANSWER)
    cache.clear(llm_string=LLM_A)
    assert _index(cache).partitions == [_partition_name(LLM_B)]
    assert cache.lookup(FRANCE, LLM_A) is None
    assert cache.lookup(FRANCE, LLM_B) == ANSWER


def test_clear_llm_string_without_a_partition_is_a_no_op():
    index = StoringFakeIndex()
    index.drop_partition = lambda name: pytest.fail(f"dropped {name}")  # type: ignore
    cache = _cache(index=index)
    cache.clear(llm_string=LLM_A)


def test_clear_drops_every_cache_partition_and_leaves_the_rest():
    cache = _cache()
    cache.update(FRANCE, LLM_A, ANSWER)
    cache.update(FRANCE, LLM_B, ANSWER)
    index = _index(cache)
    index.partitions.append("tenant_a")
    # A partition another process made for a third model is dropped too.
    index.partitions.append(PARTITION_PREFIX + "deadbeefdeadbeef")
    cache.clear()
    assert index.partitions == ["tenant_a"]
    assert cache.lookup(FRANCE, LLM_A) is None


def test_update_after_clear_recreates_the_partition():
    cache = _cache()
    cache.update(FRANCE, LLM_A, ANSWER)
    cache.clear()
    cache.update(FRANCE, LLM_A, ANSWER)
    assert _index(cache).partitions == [_partition_name(LLM_A)]
    assert cache.lookup(FRANCE, LLM_A) == ANSWER


def test_clear_rejects_unknown_keywords():
    cache = _cache()
    with pytest.raises(TypeError, match="only llm_string"):
        cache.clear(prompt=FRANCE)


# -------------------------------
# Input contracts
# -------------------------------


@pytest.mark.parametrize(
    "value",
    [True, False, float("nan"), 1.0000001, -1.0000001, "0.9", None, [0.9]],
    ids=["True", "False", "nan", "above 1", "below -1", "str", "None", "list"],
)
def test_similarity_threshold_rejects_non_cosine_values(value):
    with pytest.raises(ValueError, match="similarity_threshold"):
        _cache(value)


@pytest.mark.parametrize("value", [-1, 0, 0.9, 1, 1.0])
def test_similarity_threshold_accepts_the_cosine_range(value):
    assert _cache(value).similarity_threshold == float(value)


@pytest.mark.parametrize("prompt", [None, 3, b"bytes", [FRANCE]])
def test_prompt_must_be_a_str(prompt):
    cache = _cache()
    with pytest.raises(TypeError, match="prompt must be a str"):
        cache.lookup(prompt, LLM_A)
    with pytest.raises(TypeError, match="prompt must be a str"):
        cache.update(prompt, LLM_A, ANSWER)
    assert _index(cache).inserted == []


@pytest.mark.parametrize(
    "return_val",
    [["Paris"], [AIMessage(content="Paris")], [Generation(text="x"), {"text": "y"}]],
    ids=["str", "message", "mixed"],
)
def test_update_rejects_values_that_are_not_generations(return_val):
    cache = _cache()
    with pytest.raises(ValueError, match="Generation"):
        cache.update(FRANCE, LLM_A, return_val)
    assert _index(cache).inserted == []
    assert _index(cache).partitions == []


def test_update_accepts_a_generator_of_generations():
    cache = _cache()
    cache.update(FRANCE, LLM_A, (g for g in ANSWER))
    assert cache.lookup(FRANCE, LLM_A) == ANSWER


# -------------------------------
# Embedding
# -------------------------------


def test_vectors_are_scaled_to_unit_norm_before_storing_and_searching():
    embeddings = PromptEmbeddings({FRANCE: [3.0, 0.0, 4.0, 0.0]})
    cache = _cache(embeddings=embeddings)
    cache.update(FRANCE, LLM_A, ANSWER)
    assert _index(cache).inserted[0]["data"] == [[0.6, 0.0, 0.8, 0.0]]
    # The search vector is normalized too: a raw [3, 0, 4, 0] query would
    # score 5.0 against the stored row, so the row would hit at any threshold.
    assert cache.lookup(FRANCE, LLM_A) == ANSWER
    assert _index(cache).searched[0]["top_k"] == 4
    assert embeddings.queries == [FRANCE, FRANCE]


def test_zero_vector_is_stored_as_is_and_never_hits():
    embeddings = PromptEmbeddings({FRANCE: [0.0, 0.0, 0.0, 0.0]})
    cache = _cache(embeddings=embeddings)
    cache.update(FRANCE, LLM_A, ANSWER)
    assert _index(cache).inserted[0]["data"] == [[0.0, 0.0, 0.0, 0.0]]
    assert cache.lookup(FRANCE, LLM_A) is None


def test_chat_prompts_are_embedded_as_role_content_lines():
    prompt = dumps(
        [
            SystemMessage(content="You are terse."),
            HumanMessage(content=[{"type": "text", "text": "Capital of France?"}]),
        ]
    )
    assert embedding_text(prompt) == "system: You are terse.\nhuman: Capital of France?"


@pytest.mark.parametrize(
    "prompt",
    [
        "What is the capital of France?",
        "[not json",
        "[]",
        "[1, 2]",
        '[{"type": "constructor", "kwargs": {"content": "x"}}, "stray"]',
        json.dumps([{"lc": 1, "type": "constructor", "kwargs": {"role": "x"}}]),
        dumps([HumanMessage(content=[{"type": "image_url", "image_url": "u"}])]),
        dumps([HumanMessage(content=[{"type": "text", "text": "a"}, {"x": 1}])]),
    ],
    ids=[
        "plain text",
        "broken json",
        "empty list",
        "not messages",
        "mixed list",
        "no content",
        "image block",
        "unknown block",
    ],
)
def test_prompts_that_are_not_text_messages_are_embedded_as_is(prompt):
    assert embedding_text(prompt) == prompt


def test_embedding_uses_embed_query_on_the_reduced_text():
    prompt = dumps([HumanMessage(content=FRANCE)])
    embeddings = PromptEmbeddings({f"human: {FRANCE}": PROMPT_VECTORS[FRANCE]})
    cache = _cache(embeddings=embeddings)
    cache.update(prompt, LLM_A, ANSWER)
    assert embeddings.queries == [f"human: {FRANCE}"]
    # The row keeps the prompt LangChain gave us, not the reduced text.
    stored = json.loads(_index(cache).stored[(_partition_name(LLM_A), 1)])
    assert stored["text"] == prompt


# -------------------------------
# Async wrappers
# -------------------------------


async def test_async_methods_delegate_to_the_sync_ones():
    cache = _cache()
    await cache.aupdate(FRANCE, LLM_A, ANSWER)
    assert await cache.alookup(FRANCE_AGAIN, LLM_A) == ANSWER
    await cache.aclear(llm_string=LLM_A)
    assert await cache.alookup(FRANCE, LLM_A) is None


# -------------------------------
# Review follow-ups: fields the model reads, and partitions cleared elsewhere
# -------------------------------


@pytest.mark.parametrize(
    "messages",
    [
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "w", "args": {"city": "Paris"}, "id": "c1"}],
            )
        ],
        [ToolMessage(content="20 C", tool_call_id="c1")],
        [ChatMessage(content="hi", role="user")],
        [FunctionMessage(content="x", name="f")],
        [HumanMessage(content="hi", name="alice")],
        [
            AIMessage(
                content="",
                additional_kwargs={"function_call": {"name": "f", "arguments": "{}"}},
            )
        ],
    ],
    ids=[
        "ai tool_calls",
        "tool message",
        "chat message",
        "function message",
        "named human",
        "function_call",
    ],
)
def test_messages_with_fields_the_model_reads_are_embedded_as_is(messages):
    prompt = dumps(messages)
    assert embedding_text(prompt) == prompt


def test_ai_message_metadata_the_model_does_not_read_is_dropped():
    prompt = dumps(
        [
            HumanMessage(content="hi"),
            AIMessage(
                content="Paris",
                response_metadata={"model": "x"},
                usage_metadata={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            ),
        ]
    )
    assert embedding_text(prompt) == "human: hi\nai: Paris"


def test_tool_call_arguments_keep_conversations_apart():
    # The reviewer's case: only the tool call's argument differs, and the
    # tool's reply is the same; the two prompts must not embed alike.
    def conversation(city):
        return dumps(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "w", "args": {"city": city}, "id": "c1"}],
                ),
                ToolMessage(content="20 C", tool_call_id="c1"),
            ]
        )

    paris, berlin = conversation("Paris"), conversation("Berlin")
    assert embedding_text(paris) != embedding_text(berlin)
    assert embedding_text(paris) == paris


def _two_caches():
    index = StoringFakeIndex()
    a = _cache(index=index)
    b = _cache(index=index)
    return index, a, b


def test_update_recreates_a_partition_another_instance_cleared():
    index, a, b = _two_caches()
    a.update(FRANCE, LLM_A, ANSWER)
    b.clear()
    assert index.partitions == []
    a.update(FRANCE, LLM_A, ANSWER)
    assert index.partitions == [_partition_name(LLM_A)]
    assert a.lookup(FRANCE, LLM_A) == ANSWER
    assert b.lookup(FRANCE, LLM_A) == ANSWER


def test_update_recreates_a_partition_another_instance_cleared_by_llm_string():
    index, a, b = _two_caches()
    a.update(FRANCE, LLM_A, ANSWER)
    b.clear(llm_string=LLM_A)
    a.update(GERMANY, LLM_A, [Generation(text="Berlin")])
    assert index.partitions == [_partition_name(LLM_A)]
    assert a.lookup(FRANCE, LLM_A) is None
    assert a.lookup(GERMANY, LLM_A) == [Generation(text="Berlin")]


def test_lookup_after_another_instance_cleared_is_a_miss_not_an_error():
    index, a, b = _two_caches()
    a.update(FRANCE, LLM_A, ANSWER)
    b.clear()
    assert a.lookup(FRANCE, LLM_A) is None
    # The stale name is forgotten, so the next lookup does not even search.
    searches = len(index.searched)
    assert a.lookup(FRANCE, LLM_A) is None
    assert len(index.searched) == searches


def test_insert_failure_with_the_partition_present_is_raised():
    cache = _cache()
    cache.update(FRANCE, LLM_A, ANSWER)
    index = _index(cache)

    def failing_insert(*args, **kwargs):
        raise RuntimeError("disk full")

    index.insert = failing_insert  # type: ignore
    with pytest.raises(RuntimeError, match="disk full"):
        cache.update(GERMANY, LLM_A, ANSWER)
    assert index.partitions == [_partition_name(LLM_A)]
