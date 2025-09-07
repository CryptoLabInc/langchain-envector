# LangChain Envector Integration

Encrypted vector search for LangChain using Envector (ES2), powered by homomorphic encryption (CKKS). This repo ships a LangChain-compatible VectorStore and retriever utilities built on the high-level `es2` Python SDK.

## Table of Contents
- [Overview](#overview)
- [For Users](#for-users)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Configuration](#configuration)
  - [Data Model](#data-model)
  - [Limitations](#limitations)
  - [Troubleshooting](#troubleshooting)
- [For Contributors](#for-contributors)
  - [Development Setup](#development-setup)
  - [Testing](#testing)
  - [Guidelines](#guidelines)

## Overview
LangChain integration for encrypted vector search via Envector (ES2). The SDK performs client-side crypto so vectors and associated metadata remain encrypted end-to-end.

## For Users

## Features
- LangChain VectorStore interface (`similarity_search`, `from_texts`, etc.)
- Optional Retriever via `VectorStoreRetriever`
- End-to-end encryption handled transparently by the SDK
- Client-side filtering and score-threshold support

### Installation
- Python 3.9–3.13
- Create a virtualenv (recommended with Python 3.11):
  - `python3.11 -m venv .venv && source .venv/bin/activate`
- Install runtime dependencies:
  - `pip install -U pip setuptools wheel`
  - `pip install es2==1.0.3rc7`
  - `pip install langchain`  # VectorStore/Retriever API
  - `pip install sentence-transformers`  # or another embeddings provider

### Quick Start

With embeddings
```
from langchain.embeddings import HuggingFaceEmbeddings
from libs.envector.vectorstore import Envector
from libs.envector.config import ConnectionConfig, EnvectorConfig, IndexSettings, KeyConfig

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

cfg = EnvectorConfig(
    connection=ConnectionConfig(address="0.0.0.0:50050"),
    key=KeyConfig(key_path="./keys", key_id="example_key", preset="ip", eval_mode="rmp"),
    index=IndexSettings(index_name="demo", dim=384, query_encryption="plain"),
    create_if_missing=True,
)

store = Envector(config=cfg, embeddings=emb)
store.add_texts([
    "Homomorphic encryption enables computation over ciphertext",
    "LangChain integrates retrieval with LLMs",
], metadatas=[{"source": "note1"}, {"source": "note2"}])

docs = store.similarity_search("What computes on encrypted data?", k=2)
for d in docs:
    print(d.page_content, d.metadata)
```

Pre-computed vectors
```
from libs.envector.vectorstore import Envector
from libs.envector.config import ConnectionConfig, EnvectorConfig, IndexSettings, KeyConfig

cfg = EnvectorConfig(
    connection=ConnectionConfig(address="0.0.0.0:50050"),
    key=KeyConfig(key_path="./keys", key_id="example_key"),
    index=IndexSettings(index_name="demo", dim=16, query_encryption="cipher"),
    create_if_missing=True,
)
store = Envector(config=cfg)

texts = ["A", "B"]
metas = [{"label": "x"}, {"label": "y"}]
vectors = [[1.0] + [0.0]*15, [0.0, 1.0] + [0.0]*14]
store.add_texts(texts, metadatas=metas, vectors=vectors)

docs = store.similarity_search("q", k=1, embedding=vectors[0])
print(docs[0].page_content, docs[0].metadata)
```

### Data Model
- Envector stores a single associated data field per vector at the SDK level, called `metadata`.
- Today, this field is a raw string. To interoperate with LangChain’s `Document(page_content, metadata)`, we use a JSON envelope when inserting:
  - `{"text": <string>, "metadata": <object>}`
- On retrieval:
  - If the stored string parses as JSON and matches the envelope, we set `page_content = text` and merge the object under `metadata` with system fields (`_score`, `_id`).
  - If it is not JSON, we treat the raw string as `page_content` and only include system fields in `Document.metadata`.
- Client-side filtering works only when the JSON envelope provides an object under `metadata`.

### Configuration
- See `libs/envector/config.py` for typed configs:
  - `ConnectionConfig`, `KeyConfig`, `IndexSettings`, `EnvectorConfig`
- Important settings:
  - `connection.address` (or `host` + `port`)
  - `key.key_path`, `key.key_id`, optional `preset`, `eval_mode`
  - `index.index_name`, `index.dim` (16–4096), `index.query_encryption` (`plain|cipher`)
  - `index.output_fields` (defaults to `["metadata"]`), `index.fetch_k` for over-fetching
  - `create_if_missing` (bool)

### Retriever
- `retriever = store.as_retriever(search_kwargs={"k": 3})` integrates with LangChain chains for RAG.

## For Contributors

### Development Setup
- Recommended: Python 3.11 virtualenv
  - `python3.11 -m venv .venv && source .venv/bin/activate`
- Install editable and tools
  - `pip install -U pip setuptools wheel`
  - `pip install -e .`
  - `pip install -r tests/requirements.txt`

### Testing
- Unit tests (no server) using fakes:
  - `python run_unit_tests.py`
- Integration tests (require ES2 server and keys):
  - Set env: `ES2_ADDRESS`, `ES2_KEY_PATH`, `ES2_KEY_ID`
  - Optional: `ES2_USE_EMBEDDINGS=1`, `ES2_EMB_MODEL`, `ES2_USE_HF_DATASET=1`, dataset params
  - Run: `pytest -m integration -s`

### Guidelines
- Keep code/comments/docs in English.
- Use high-level `es2` SDK APIs; avoid low-level gRPC/Indexer.
- Follow existing style and keep changes minimal and well-scoped.

## Limitations
- Item-level delete/update is not provided by the SDK (drop index only)
- Manual item IDs are not supported (returned IDs from `add_texts` are ephemeral)
- Filtering is client-side and requires JSON envelope with `metadata` object

### Troubleshooting
- Connection: verify server address and keys are correct/registered.
- Embeddings: ensure model dimension matches `index.dim` if inserting vectors directly.
- Results: if you see raw strings as `page_content`, confirm inserts used the JSON envelope.
