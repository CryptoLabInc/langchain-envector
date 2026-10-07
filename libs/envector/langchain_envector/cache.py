"""Semantic LLM cache backed by an enVector index.

`EnvectorSemanticCache` implements LangChain's ``BaseCache`` so that a prompt
close enough to one answered before is served from the cache instead of the
model. Prompts and the cached generations live in an enVector index: the
prompt embeddings are searched under encryption, and the prompt text and the
generations are AES-encrypted when the index has ``metadata_encryption`` on.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import numbers
import threading
import warnings
from typing import Any, Dict, List, Optional, Sequence, Set

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.load import dumps, loads
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    Generation,
    GenerationChunk,
)

from .config import EnvectorConfig
from .types import Embeddings, as_embeddings
from .vectorstore import Envector, _is_empty_shard_list_error

# Every partition this cache creates starts with this, so `clear()` can find
# them all with `list_partitions()` and leave the rest of the index alone.
PARTITION_PREFIX = "lc_cache_"

# A lookup needs the single nearest row, but the server occasionally pads a
# result with an empty placeholder hit, which the store drops. Asking for a
# few more rows keeps the real nearest row in the answer.
_LOOKUP_K = 4

# The classes a cached ``return_val`` may deserialize into: what `update`
# wrote, nothing else. ``loads`` grew ``allowed_objects`` after the oldest
# langchain-core this package accepts, so it is passed only where it exists.
_CACHED_CLASSES = [
    Generation,
    ChatGeneration,
    GenerationChunk,
    ChatGenerationChunk,
    AIMessage,
    AIMessageChunk,
]
_LOADS_TAKES_ALLOWED_OBJECTS = "allowed_objects" in inspect.signature(loads).parameters


def _partition_name(llm_string: str) -> str:
    """The partition that holds the rows cached for ``llm_string``.

    ``llm_string`` is the model's serialized configuration, often hundreds of
    characters of JSON, so the partition is named after its hash. The stored
    rows carry the full ``llm_string`` and `lookup` checks it, so a hash
    collision costs a miss, never a wrong answer.
    """
    digest = hashlib.sha256(llm_string.encode("utf-8")).hexdigest()
    return PARTITION_PREFIX + digest[:16]


def _unit(vector: Sequence[float]) -> List[float]:
    """``vector`` scaled to unit norm, so inner products are cosine similarities.

    A zero vector cannot be scaled and is returned as is: its scores are all
    zero, which is a miss at any threshold above zero.
    """
    values = [float(x) for x in vector]
    norm = math.sqrt(sum(x * x for x in values))
    if norm == 0.0 or not math.isfinite(norm):
        return values
    return [x / norm for x in values]


def _content_text(content: Any) -> Optional[str]:
    """The text of a message's ``content``, or ``None`` when it is not text only.

    Content is a string, or a list of strings and ``{"type": "text", ...}``
    blocks. Any other block (an image, a document) returns ``None`` so the
    caller falls back to embedding the whole serialized prompt: two prompts
    that differ only in such a block must not look identical.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            parts.append(block["text"])
        else:
            return None
    return "\n".join(parts)


# The message kinds `embedding_text` reduces to a ``role: content`` line: the
# ones whose role is fixed by their type. A ``ChatMessage`` carries its role in
# a separate field, a ``ToolMessage`` answers a specific tool call, and a
# ``FunctionMessage`` is named after its function; all of those stay raw.
_PLAIN_MESSAGE_TYPES = {"human", "system", "ai"}

# Serialized message fields the model does not read, so they may differ
# between two prompts that are the same conversation.
_IGNORED_MESSAGE_KEYS = {
    "content",
    "type",
    "id",
    "response_metadata",
    "usage_metadata",
    "example",
}

# Fields the model does read, but only when they hold something. LangChain
# serializes them as empty for a plain message.
_EMPTY_ONLY_MESSAGE_KEYS = {"tool_calls", "invalid_tool_calls", "additional_kwargs"}


def _plain_message_line(message: Any) -> Optional[str]:
    """``role: content`` for a serialized human, system or AI message with
    text-only content and nothing else the model reads; ``None`` otherwise.

    A tool call on an AI message, a ``function_call`` in ``additional_kwargs``,
    a speaker ``name``, a ``ChatMessage`` role, a ``ToolMessage``'s
    ``tool_call_id`` — any field the model reads besides the text — returns
    ``None``: two conversations that differ only there must not reduce to the
    same text.
    """
    if not isinstance(message, dict) or message.get("type") != "constructor":
        return None
    kwargs = message.get("kwargs")
    if not isinstance(kwargs, dict) or "content" not in kwargs:
        return None
    role = kwargs.get("type")
    if role not in _PLAIN_MESSAGE_TYPES:
        return None
    for key, value in kwargs.items():
        if key in _IGNORED_MESSAGE_KEYS:
            continue
        if key in _EMPTY_ONLY_MESSAGE_KEYS and not value:
            continue
        return None
    text = _content_text(kwargs["content"])
    if text is None:
        return None
    return f"{role}: {text}"


def embedding_text(prompt: str) -> str:
    """The text that is embedded for ``prompt``.

    A chat model hands the cache its messages serialized as JSON, each wrapped
    in LangChain's ``{"lc": 1, "type": "constructor", ...}`` envelope. Embedding
    that JSON makes every prompt look alike — the envelopes and the shared
    system message dominate — so this strips it down to one ``role: content``
    line per message. That happens only when every message is a human, system
    or AI message with text-only content and nothing else the model reads
    (see `_plain_message_line`); a prompt with a tool call, a tool result, a
    ``ChatMessage`` role, a speaker name or an image block, or that is not
    such a list at all, is embedded as it is.

    The row stores ``prompt`` itself either way; this only shapes what the
    embedding model sees.
    """
    if not prompt.lstrip().startswith("["):
        return prompt
    try:
        messages = json.loads(prompt)
    except ValueError:
        return prompt
    if not isinstance(messages, list) or not messages:
        return prompt
    lines: List[str] = []
    for message in messages:
        line = _plain_message_line(message)
        if line is None:
            return prompt
        lines.append(line)
    return "\n".join(lines)


def _check_threshold(value: Any) -> float:
    """``value`` as the cosine-similarity threshold it names, or ``ValueError``.

    Accepts a real number in [-1, 1]; refuses ``bool``, NaN, strings and
    anything else rather than letting ``float()`` guess.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            "similarity_threshold must be a number between -1 and 1 "
            f"(a cosine similarity); got {value!r}."
        )
    threshold = float(value)
    if math.isnan(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError(
            "similarity_threshold must be between -1 and 1 "
            f"(a cosine similarity); got {value!r}."
        )
    return threshold


def _check_prompt(prompt: Any, label: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError(
            f"EnvectorSemanticCache.{label}: prompt must be a str; "
            f"got {type(prompt).__name__}."
        )
    return prompt


class EnvectorSemanticCache(BaseCache):
    """LangChain LLM cache that matches prompts by meaning, stored in enVector.

    Usage::

        from langchain_core.globals import set_llm_cache

        set_llm_cache(
            EnvectorSemanticCache(
                config=cfg, embeddings=emb, similarity_threshold=0.9
            )
        )

    ``config`` and ``embeddings`` are what `Envector` takes. Set
    ``IndexSettings.metadata_encryption=True`` to keep prompts and generations
    encrypted at rest; the embeddings are always encrypted.

    How it works:

    - Each distinct ``llm_string`` (the model and its invocation parameters,
      serialized by LangChain) gets its own partition, named after a hash of
      it, so a lookup only ever sees rows cached for the same model and
      settings, and ``clear(llm_string=...)`` is one ``drop_partition``.
    - `lookup` embeds the prompt (see `embedding_text`), scales the vector to
      unit norm, and takes the nearest row whose inner product — a cosine
      similarity — is at least ``similarity_threshold``. ``1.0`` is an
      identical prompt. A row stored under another ``llm_string`` is never
      returned.
    - `update` appends a row. It does not look for an earlier row for the
      same prompt: LangChain calls `update` only after a `lookup` missed, so
      there was no row within the threshold to replace. Two processes that
      answer the same prompt at once leave two rows, and `lookup` returns the
      nearest.
    - `clear()` drops every partition whose name starts with
      `PARTITION_PREFIX`; ``clear(llm_string=...)`` drops that one. Rows in
      other partitions of the index are left alone.

    The async methods run these in a thread, as ``BaseCache`` does by default.
    """

    def __init__(
        self,
        *,
        config: EnvectorConfig,
        embeddings: Embeddings,
        similarity_threshold: float = 0.9,
        client: Optional[Any] = None,
    ) -> None:
        """``similarity_threshold`` is the cosine similarity a stored prompt
        needs to count as a hit, in [-1, 1]; higher is stricter.

        Chat prompts that share a long system message score high against each
        other, so raise the threshold when that is the case. ``client`` is the
        `EnvectorClient` to use instead of one built from ``config``; tests
        pass a fake here.
        """
        self.similarity_threshold = _check_threshold(similarity_threshold)
        self._embeddings = as_embeddings(embeddings)
        # The store gets no embeddings: this cache embeds and normalizes
        # itself and hands the store vectors.
        self.vectorstore = Envector(config=config, client=client)
        self._known_partitions: Set[str] = set()
        self._lock = threading.Lock()

    # -------------------------------
    # BaseCache API
    # -------------------------------
    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """The generations cached for a prompt similar to ``prompt`` under
        ``llm_string``, or ``None``.

        ``prompt`` must be a ``str``. A hit needs a row in the partition for
        ``llm_string`` scoring at least ``similarity_threshold`` and carrying
        that same ``llm_string``. A row whose stored generations cannot be
        deserialized is reported with a ``UserWarning`` and counts as a miss.
        """
        _check_prompt(prompt, "lookup")
        partition = _partition_name(llm_string)
        if not self._partition_exists(partition):
            return None
        vector = self._embed(prompt)
        try:
            hits = self.vectorstore.similarity_search_with_score_by_vector(
                vector, k=_LOOKUP_K, fetch_k=_LOOKUP_K, partition_names=[partition]
            )
        except Exception as e:  # narrow-matched below, re-raised otherwise
            # Another cache on the same index may have cleared this partition
            # since this one last saw it: then there is nothing to find.
            if self._partition_gone(partition):
                return None
            # The store turns "no shards" into an empty result only when the
            # whole index is empty; a partition with no rows yet gets the same
            # answer from the server, so check that partition instead.
            if _is_empty_shard_list_error(e) and self._partition_rows(partition) == 0:
                return None
            raise
        for doc, score in hits:
            if score < self.similarity_threshold:
                # Hits are ranked, so nothing after this one passes either.
                return None
            if doc.metadata.get("llm_string") != llm_string:
                continue
            generations = self._cached_generations(doc.metadata.get("return_val"))
            if generations is None:
                warnings.warn(
                    f"EnvectorSemanticCache: row {doc.id} in partition {partition} "
                    "holds generations that cannot be deserialized; treating it "
                    "as a miss. Delete the row or clear the cache for this "
                    "llm_string.",
                    UserWarning,
                    stacklevel=2,
                )
                return None
            return generations
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Cache ``return_val`` for ``prompt`` under ``llm_string``.

        ``prompt`` must be a ``str`` and every entry of ``return_val`` a
        ``Generation`` (``ChatGeneration`` included); anything else raises
        before the server is contacted. Appends a row; see the class docstring
        for why an earlier row for the same prompt is not replaced.
        """
        _check_prompt(prompt, "update")
        generations = list(return_val)
        for gen in generations:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "EnvectorSemanticCache only caches Generation values; "
                    f"got {type(gen).__name__}."
                )
        partition = _partition_name(llm_string)
        self._ensure_partition(partition)
        vector = self._embed(prompt)
        metadata = {"llm_string": llm_string, "return_val": dumps(generations)}
        try:
            self.vectorstore.add_texts(
                [prompt],
                metadatas=[metadata],
                vectors=[vector],
                partition_name=partition,
            )
        except Exception:
            # Another cache on the same index may have cleared this partition
            # since this one created it; the server then refuses the insert.
            # Recreate it once and retry; any other failure is re-raised.
            if not self._partition_gone(partition):
                raise
            self._ensure_partition(partition)
            self.vectorstore.add_texts(
                [prompt],
                metadatas=[metadata],
                vectors=[vector],
                partition_name=partition,
            )

    def clear(self, **kwargs: Any) -> None:
        """Drop cached rows.

        ``clear()`` drops every partition of the index named with
        `PARTITION_PREFIX`, including ones other processes created.
        ``clear(llm_string=...)`` drops the partition for that ``llm_string``
        and is a no-op when there is none. Any other keyword raises
        ``TypeError`` rather than being ignored.
        """
        llm_string = kwargs.pop("llm_string", None)
        if kwargs:
            raise TypeError(
                "EnvectorSemanticCache.clear takes only llm_string; "
                f"got {sorted(kwargs)}."
            )
        if llm_string is not None:
            partition = _partition_name(llm_string)
            names = [partition] if self._partition_exists(partition) else []
        else:
            names = [
                name
                for name in self._server_partitions()
                if name.startswith(PARTITION_PREFIX)
            ]
        for name in names:
            self.vectorstore.drop_partition(name)
            with self._lock:
                self._known_partitions.discard(name)

    # -------------------------------
    # Helpers
    # -------------------------------
    def _embed(self, prompt: str) -> List[float]:
        # Prompts are compared with prompts, so both sides use `embed_query`:
        # a model with distinct query and passage encodings must not put the
        # stored and the looked-up prompt in different spaces.
        return _unit(self._embeddings.embed_query(embedding_text(prompt)))

    def _server_partitions(self) -> Dict[str, Any]:
        """Partition name -> its ``list_partitions`` entry."""
        entries = self.vectorstore.list_partitions() or []
        return {str(p.get("name")): p for p in entries if isinstance(p, dict)}

    def _partition_exists(self, partition: str) -> bool:
        with self._lock:
            if partition in self._known_partitions:
                return True
        if partition in self._server_partitions():
            with self._lock:
                self._known_partitions.add(partition)
            return True
        return False

    def _partition_gone(self, partition: str) -> bool:
        """True when the server no longer lists ``partition``; forgets it then.

        The in-process set of known partitions is a shortcut, not proof: a
        `clear` from another cache instance on the same index drops the
        partition without telling this one. Callers consult this after the
        server refused a search or an insert.
        """
        if partition in self._server_partitions():
            return False
        with self._lock:
            self._known_partitions.discard(partition)
        return True

    def _partition_rows(self, partition: str) -> int:
        """Rows in ``partition`` as the server reports them; 0 when it is gone."""
        entry = self._server_partitions().get(partition)
        if entry is None:
            return 0
        try:
            return int(entry.get("num_vectors", 0))
        except (TypeError, ValueError):
            return 0

    def _ensure_partition(self, partition: str) -> None:
        """Create ``partition`` unless it exists. Serialized within the process;
        another process creating the same name at the same time makes
        ``create_partition`` fail, which is fine once the partition is there."""
        with self._lock:
            if partition in self._known_partitions:
                return
            if partition not in self._server_partitions():
                try:
                    self.vectorstore.create_partition(partition)
                except Exception:
                    if partition not in self._server_partitions():
                        raise
            self._known_partitions.add(partition)

    @staticmethod
    def _cached_generations(raw: Any) -> Optional[List[Generation]]:
        """The generations serialized in ``raw``, or ``None`` when ``raw`` is
        not what `update` writes."""
        if not isinstance(raw, str):
            return None
        try:
            if _LOADS_TAKES_ALLOWED_OBJECTS:
                value = loads(raw, allowed_objects=_CACHED_CLASSES)
            else:
                value = loads(raw)
        except Exception:
            return None
        if not isinstance(value, list) or not all(
            isinstance(g, Generation) for g in value
        ):
            return None
        return value
