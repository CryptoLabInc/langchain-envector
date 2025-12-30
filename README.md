# LangChain Envector Integration

Encrypted vector search for LangChain using Envector, powered by homomorphic encryption (CKKS). This repo ships a LangChain-compatible VectorStore and retriever utilities built on the high-level `pyenvector` Python SDK.

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
  - `pip install pyenvector langchain sentence-transformers`

## Usage Overview
1. Configure Envector using `EnvectorConfig`, pointing to your EnVector endpoint and keys.
2. Initialize embeddings (or provide pre-computed vectors).
3. Instantiate `Envector(config=cfg, embeddings=emb)` and call `add_texts`, `add_documents`, or use `as_retriever`.
4. Run `similarity_search` or plug the retriever into your LangChain pipeline.

> See `notebooks/` for end-to-end walkthroughs and the `libs/envector` package for implementation details.

## Configuration
Key dataclasses live in `libs/envector/config.py`:
- `ConnectionConfig`: address or host/port for EnVector.
- `KeyConfig`: key path, key ID, optional preset/eval mode.
- `IndexSettings`: index name, dimension (32–4096), query encryption mode, optional output fields and fetch parameters.
- `EnvectorConfig`: wraps the above and enables auto-creation via `create_if_missing`.

## Data Model
- Each vector stores a single `metadata` string in EnVector.
- To align with LangChain’s `Document`, inserts wrap data as JSON: `{"text": ..., "metadata": ...}`.
- Retrieval unwraps JSON, returning `Document(page_content=text, metadata={...})`.
- Client-side filtering requires the JSON envelope to include an object under `metadata`.

## Limitations
- Item-level delete/update is unsupported (drop the index to reset).
- Manual item IDs are not accepted; returned IDs from `add_texts` are ephemeral.
- Filtering happens client-side; ensure metadata is JSON for structured filters.

## Examples
### Configuration

```python
  from langchain_envector.config import ConnectionConfig, EnvectorConfig, IndexSettings, KeyConfig

  cfg = EnvectorConfig(
      connection=ConnectionConfig(
        address=ENVECTOR_ADDRESS, 
        access_token=ENVECTOR_ACCESS_TOKEN
      ),
      key=KeyConfig(
        key_path=ENVECTOR_KEY_PATH, 
        key_id=ENVECTOR_KEY_ID, 
        preset="ip", 
        eval_mode="rmp"
      ),
      index=IndexSettings(
        index_name=INDEX_NAME, 
        dim=vector_dim, 
        query_encryption="cipher"
      ),
      create_if_missing=True,
  )
  ```

### Add documents (from LangChain Documents):

```python
from langchain_core.documents import Document
from langchain_envector.vectorstore import Envector

docs = [
  Document(
    page_content="chunk-1", 
    metadata={"source": "paper.pdf", "page": 1, "chunk": 0}
  ),
  Document(
    page_content="chunk-2", 
    metadata={"source": "paper.pdf", "page": 1, "chunk": 1}
  ),
]

store = Envector(config=cfg, embeddings=emb)
store.add_documents(docs)
```

Or you can use `add_texts` to store vectors and their texts.

```python
store.add_texts(
    texts=["chunk 4"],
    metadatas=[{"source": "paper.pdf", "page": 1, "chunk": 4}]
)
```

### Similarity search

```python
results = store.similarity_search(query, k=1)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")
```

#### Similarity Search with Score

```python
results = store.similarity_search_with_score(query, k=1)
for doc, score in results:
    print(f"* [SIM={score:.3f}] {doc.page_content} [{doc.metadata}]")
```


#### Similarity Search with Vector

```python
query_embedding = embeddings.embed_query(query)
print(f"Query: {query_embedding[:3]}")
results = store.similarity_search_by_vector(query_embedding, k=3)
for doc in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
```


## Troubleshooting
- Connection issues: verify EnVector address and registered keys.
- Embeddings mismatch: ensure embedding dimension equals `index.dim` when supplying vectors.
- Unexpected raw strings: confirm inserts used the JSON envelope.
- Key Issues: check key's metadata to sync with the registered key if facing any key issue.

## Testing Without EnVector
- Run unit tests offline (no EnVector or SDK required):
  - `python -m pytest -q -m "not integration"`
  - or `python scripts/run_unit_tests.py`
- Run integration tests (requires server and keys):
  - Export environment variables: `ENVECTOR_ADDRESS`, `ENVECTOR_KEY_PATH`, `ENVECTOR_KEY_ID`, and `ENVECTOR_INDEX_NAME`.
  - Optional: `ENVECTOR_USE_EMBEDDINGS=1`, `ENVECTOR_EMB_MODEL`, `ENVECTOR_USE_HF_DATASET=1`
  - `python -m pytest -q -m integration -s`

## Contributing
See [`CONTRIBUTE.md`](CONTRIBUTE.md) for development, testing, and PR guidelines.
