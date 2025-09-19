from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union, overload


class Embeddings(Protocol):
    """Minimal Embeddings protocol to avoid hard-coding a dependency on LangChain.

    LangChain-compatible embeddings typically implement these two methods.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover - interface only
        ...

    def embed_query(self, text: str) -> List[float]:  # pragma: no cover - interface only
        ...


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: Dict[str, Any]


def pack_metadata(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Pack text and metadata into a single JSON string field accepted by ES2.

    ES2 metadata API stores lists of strings; we store a single JSON blob per item.
    Item-level IDs are not persisted/addressable.
    """
    import json

    payload = {
        "text": text,
        "metadata": metadata or {},
    }
    return json.dumps(payload, ensure_ascii=False)


def unpack_metadata(raw: Any) -> Dict[str, Any]:
    """Return metadata as a dict regardless of the raw payload type.

    Recent ES2 versions may return decrypted metadata as a Python dict instead
    of the JSON string we originally stored. We normalise the payload here so
    downstream code always works with a dictionary.
    """

    import json

    # Already a dict → nothing to do.
    if isinstance(raw, dict):
        return raw

    print("slafjklshglkhslafhlksadjlghsal;hf")

    # Some responses wrap the payload in a single-element list.
    if isinstance(raw, (list, tuple)):
        if len(raw) == 1:
            return unpack_metadata(raw[0])
        return {"_raw": list(raw)}

    if raw is None:
        return {"_raw": None}

    # Decode bytes before attempting JSON parsing.
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return {"_raw": raw}

    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            # Some ES2 responses return Python-literal strings (single quotes).
            try:
                import ast

                data = ast.literal_eval(raw)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {"_raw": raw}

    # Fallback: expose the raw object for debugging purposes.
    return {"_raw": raw}


# --- Embeddings adaptation helpers -----------------------------------------------------

class _CallableEmbeddings:
    def __init__(self, docs_fn: Callable[[List[str]], List[List[float]]], query_fn: Callable[[str], List[float]]):
        self._docs_fn = docs_fn
        self._query_fn = query_fn

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._docs_fn(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._query_fn(text)


def as_embeddings(emb: Any) -> Embeddings:
    """Adapt various embedding providers into the Embeddings protocol.

    Supported inputs:
    - Object with `embed_documents` and `embed_query` (LangChain-style) — returned as-is.
    - Object with `encode` supporting list[str] -> list[vec] and str -> vec — wrapped.
    - Tuple(callable_docs, callable_query) — wrapped.
    """
    # Case 1: Already satisfies the protocol
    if hasattr(emb, "embed_documents") and hasattr(emb, "embed_query"):
        return emb  # type: ignore[return-value]

    # Case 2: Sentence-Transformers style `.encode()`
    if hasattr(emb, "encode"):
        encode = getattr(emb, "encode")

        def docs_fn(texts: List[str]) -> List[List[float]]:
            return list(encode(texts))

        def query_fn(text: str) -> List[float]:
            out = encode([text])
            return list(out[0]) if isinstance(out, (list, tuple)) else list(out)

        return _CallableEmbeddings(docs_fn, query_fn)

    # Case 3: Tuple of callables
    if isinstance(emb, tuple) and len(emb) == 2 and callable(emb[0]) and callable(emb[1]):
        docs_fn, query_fn = emb  # type: ignore[assignment]
        return _CallableEmbeddings(docs_fn, query_fn)

    raise TypeError(
        "Unsupported embeddings provider. Provide an object with 'embed_documents'/'embed_query', "
        "an object with 'encode', or a tuple of two callables (docs_fn, query_fn)."
    )
