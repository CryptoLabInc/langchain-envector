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
  - `pip install es2==1.1.0rc1 langchain sentence-transformers`

## Usage Overview
1. Configure Envector using `EnvectorConfig`, pointing to your ES2 endpoint and keys.
2. Initialize embeddings (or provide pre-computed vectors).
3. Instantiate `Envector(config=cfg, embeddings=emb)` and call `add_texts` or `as_retriever`.
4. Run `similarity_search` or plug the retriever into your LangChain pipeline.

> See `notebooks/` for end-to-end walkthroughs and the `libs/envector` package for implementation details.

## Configuration
Key dataclasses live in `libs/envector/config.py`:
- `ConnectionConfig`: address or host/port for ES2.
- `KeyConfig`: key path, key ID, optional preset/eval mode.
- `IndexSettings`: index name, dimension (16–4096), query encryption mode, optional output fields and fetch parameters.
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

## Troubleshooting
- Connection issues: verify ES2 address and registered keys.
- Embeddings mismatch: ensure embedding dimension equals `index.dim` when supplying vectors.
- Unexpected raw strings: confirm inserts used the JSON envelope.

## Contributing
See [`CONTRIBUTE.md`](CONTRIBUTE.md) for development, testing, and PR guidelines.
