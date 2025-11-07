# LangChain Envector Integration

Encrypted vector search for LangChain using Envector (ES2), powered by homomorphic encryption (CKKS). This repo ships a LangChain-compatible VectorStore and retriever utilities built on the high-level `es2` Python SDK.

## Features
- LangChain `VectorStore` interface with `similarity_search`, `from_texts`, etc.
- Optional `VectorStoreRetriever` helper for quick RAG integrations.
- Client-side encryption handled transparently by the SDK, including score thresholds and filtering.

## Installation
- Python 3.9–3.13 (recommend 3.11)
- Create and activate a virtualenv:
  - `python3.11 -m venv .venv && source .venv/bin/activate`
- Install runtime dependencies:
  - `pip install -U pip setuptools wheel`
  - `pip install es2 langchain sentence-transformers`

## Usage Overview
1. Configure Envector using `EnvectorConfig`, pointing to your ES2 endpoint and keys.
2. Initialize embeddings (or provide pre-computed vectors).
3. Instantiate `Envector(config=cfg, embeddings=emb)` and call `add_texts`, `add_documents`, or use `as_retriever`.
4. Run `similarity_search` or plug the retriever into your LangChain pipeline.

> See `notebooks/` for end-to-end walkthroughs and the `libs/envector` package for implementation details.

## Configuration
Key dataclasses live in `libs/envector/config.py`:
- `ConnectionConfig`: address or host/port for ES2.
- `KeyConfig`: key path, key ID, optional preset/eval mode.
- `IndexSettings`: index name, dimension (32–4096), query encryption mode, optional output fields and fetch parameters.
- `EnvectorConfig`: wraps the above and enables auto-creation via `create_if_missing`.

## Data Model
- Each vector stores a single `metadata` string in ES2.
- To align with LangChain’s `Document`, inserts wrap data as JSON: `{"text": ..., "metadata": ...}`.
- Retrieval unwraps JSON, returning `Document(page_content=text, metadata={...})`.
- Client-side filtering requires the JSON envelope to include an object under `metadata`.

## Limitations
- Item-level delete/update is unsupported (drop the index to reset).
- Manual item IDs are not accepted; returned IDs from `add_texts` are ephemeral.
- Filtering happens client-side; ensure metadata is JSON for structured filters.

## Examples
- Configuration
  ```python
  from langchain_envector.config import ConnectionConfig, EnvectorConfig, IndexSettings, KeyConfig

  cfg = EnvectorConfig(
      connection=ConnectionConfig(address=ES2_ADDRESS, access_token=ES2_ACCESS_TOKEN) if ES2_ACCESS_TOKEN else ConnectionConfig(address=ES2_ADDRESS),
      key=KeyConfig(key_path=ES2_KEY_PATH, key_id=ES2_KEY_ID, preset="ip", eval_mode="rmp"),
      index=IndexSettings(index_name=INDEX_NAME, dim=vector_dim, query_encryption="cipher"),
      create_if_missing=True,
  )
  ```

- Add documents (from LangChain Documents):

  ```python
  from langchain_core.documents import Document
  from langchain_envector.vectorstore import Envector

  docs = [
    Document(page_content="chunk-1", metadata={"source": "paper.pdf", "page": 1, "chunk": 0}),
    Document(page_content="chunk-2", metadata={"source": "paper.pdf", "page": 1, "chunk": 1}),
  ]
  
  store = Envector(config=cfg, embeddings=emb)
  store.add_documents(docs)
  ```

## Troubleshooting
- Connection issues: verify ES2 address and registered keys.
- Embeddings mismatch: ensure embedding dimension equals `index.dim` when supplying vectors.
- Unexpected raw strings: confirm inserts used the JSON envelope.

## Testing Without ES2
- Run unit tests offline (no ES2 or SDK required):
  - `python -m pytest -q -m "not integration"`
  - or `python run_unit_tests.py`
- Run integration tests (requires server and keys):
  - Export `ES2_ADDRESS`, `ES2_KEY_PATH`, `ES2_KEY_ID`
  - Optional: `ES2_USE_EMBEDDINGS=1`, `ES2_EMB_MODEL`, `ES2_USE_HF_DATASET=1`
  - `python -m pytest -q -m integration -s`

## Contributing
See [`CONTRIBUTE.md`](CONTRIBUTE.md) for development, testing, and PR guidelines.
