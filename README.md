# LangChain Envector Integration

Encrypted vector search for LangChain using Envector, powered by homomorphic encryption (CKKS). This repo ships a LangChain-compatible VectorStore and retriever utilities built on the high-level `pyenvector` Python SDK.

## Features
- LangChain `VectorStore` interface with `similarity_search`, `from_texts`, etc.
- Optional `VectorStoreRetriever` helper for quick RAG integrations.
- Client-side encryption handled transparently by the SDK, including score thresholds and filtering.
- In-place `delete`, `update_documents` and `upsert_documents` by item ID, plus named partitions.

Requires `pyenvector >= 1.6.0rc1`.

## Installation
- Python 3.9–3.13 (recommend 3.11)
- Create and activate a virtualenv:
  - `python3.11 -m venv .venv && source .venv/bin/activate`
- Install runtime dependencies:
  - `pip install -U pip setuptools wheel`
  - `pip install 'pyenvector>=1.6.0rc1' langchain sentence-transformers`

## Usage Overview
1. Configure Envector using `EnvectorConfig`, pointing to your EnVector endpoint and keys.
2. Initialize embeddings (or provide pre-computed vectors).
3. Instantiate `Envector(config=cfg, embeddings=emb)` and call `add_texts`, `add_documents`, or use `as_retriever`.
4. Run `similarity_search` or plug the retriever into your LangChain pipeline.

> See `notebooks/` for end-to-end walkthroughs and the `libs/envector` package for implementation details.

## Configuration
Key dataclasses live in `libs/envector/config.py`:
- `ConnectionConfig`: address or host/port for EnVector; optional `kms_address` / `kms_secure` / `kms_ca_cert` for the enVector KMS service. When `kms_address` is set, keys are KMS-managed — omit `KeyConfig.key_path`.
- `KeyConfig`: key path, key ID, optional preset/eval mode.
- `IndexSettings`: index name, dimension (32–4096), query encryption mode, optional output fields and fetch parameters.
- `WriteSettings`: whether each write path waits for the server before returning. EnVector writes are asynchronous server-side, but what that means for the next read differs per operation: inserts are searchable immediately and do not wait, while updates return before the rebuilt rows are visible and do wait. Each default is documented with the measurement behind it.
- `EnvectorConfig`: wraps the above and enables auto-creation via `create_if_missing`.

## Data Model
- Each vector stores a single `metadata` string in EnVector.
- To align with LangChain’s `Document`, inserts wrap data as JSON: `{"text": ..., "metadata": ...}`.
- Retrieval unwraps JSON, returning `Document(page_content=text, metadata={...})`.
- Client-side filtering requires the JSON envelope to include an object under `metadata`.

## Limitations
- Manual item IDs are not accepted on insert: EnVector issues its own `item_id` values and cannot create an item under a caller-chosen ID. Use the returned IDs for subsequent `delete` / `update_documents` / `upsert_documents` calls.
- Fetch-by-ID (`get_by_ids`) is unsupported.
- Filtering happens client-side; ensure metadata is JSON for structured filters.
- Multi-key indexes and cloud key stores are not wired up yet — see [`TODO.md`](TODO.md).
- One enVector endpoint per process. `pyenvector` keeps a single process-wide connection, so several stores can coexist only while they all point at the same endpoint; the integration reuses that connection for them. Two stores pointing at **different** servers in one process is not supported — the second connection closes the first one's channel.
- `update_documents` / `upsert_documents` calls larger than 10,000 items are split into several server transactions. If a later chunk fails, the earlier ones stay applied.
- The first `update_documents` / `upsert_documents` after an un-awaited `add_texts` blocks until those inserts have merged (~15s), because mutating an unmerged row drops it from search — see [`TODO.md`](TODO.md). Inserts themselves stay fast, and search and delete never wait.

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
        preset="ip3", 
        eval_mode="mms32"
      ),
      index=IndexSettings(
        index_name=INDEX_NAME, 
        dim=vector_dim, 
        query_encryption="plain"
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
    texts=["chunk 3"],
    metadatas=[{"source": "paper.pdf", "page": 1, "chunk": 2}]
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

### Update existing items

`add_texts` / `add_documents` return the `item_id` values EnVector assigned. Pass
them back to replace an item in place, preserving its ID (requires pyenvector >= 1.6.0):

```python
ids = store.add_texts(["draft"], metadatas=[{"status": "draft"}])

# Replace both the vector and the stored payload: page_content is re-embedded.
result = store.update_documents(
    ids, [Document(page_content="final", metadata={"status": "final"})]
)
print(result)  # {"request_id": [...], "not_found_item_ids": [...]}
```

IDs that match no live row (missing or already deleted) come back in
`not_found_item_ids` rather than raising.

For a metadata-only change that leaves the vector — and therefore what the item
matches — untouched, use `update_metadata`, or
`update_documents(..., update_vectors=False)`:

```python
store.update_metadata(ids, ["final"], metadatas=[{"status": "final"}])
```

### Insert and update in one call

`upsert_documents` routes each document by whether it carries an ID: `None`
inserts, an existing `item_id` replaces in place. Note that a caller-chosen ID
cannot create a new item — an ID matching no live row is reported in
`not_found_item_ids`.

```python
result = store.upsert_documents(
    [Document(page_content="revised"), Document(page_content="brand new")],
    ids=[ids[0], None],
)
print(result["inserted_item_ids"])  # IDs issued for the ID-less entries
```

### Delete

```python
store.delete(ids)  # accepts ints or numeric strings, e.g. doc.id
```

Deletion is asynchronous server-side; by default this waits until the affected
shards are rebuilt, which is the SDK's own default and returned immediately in
measurement.

### Partitions

Named partitions isolate subsets of an index:

```python
store.create_partition("tenant_a")

store.add_texts(["tenant-a data"], partition_name="tenant_a")
results = store.similarity_search(query, k=3, partition_names=["tenant_a"])

print(store.list_partitions())  # [{"name": ..., "status": ..., "num_vectors": ...}]
store.drop_partition("tenant_a")  # removes the partition and its data
```

Omitting `partition_name` / `partition_names` uses the default partition or searches the whole index.


## Troubleshooting
- Connection issues: verify EnVector address and registered keys.
- Embeddings mismatch: ensure embedding dimension equals `index.dim` when supplying vectors.
- Unexpected raw strings: confirm inserts used the JSON envelope.
- Key Issues: check key's metadata to sync with the registered key if facing any key issue.

## Test

Before running tests, install dependencies for pytest:

```bash
pip install -r tests/requirements.txt
```

### Unit Test

Run unit tests offline (no EnVector or SDK required)

```bash
python -m pytest -q -m "not integration"
# or
python scripts/run_unit_tests.py
```

### Integration Test

Run integration tests (requires enVector server)

1. Prepare the running enVector server

2. Export the environment variables:

  - `ENVECTOR_ADDRESS`
  - `ENVECTOR_KEY_PATH`
  - `ENVECTOR_KEY_ID`
  - `ENVECTOR_INDEX_NAME`
  - (Optional) `ENVECTOR_USE_EMBEDDINGS=1`
  - (Optional) `ENVECTOR_EMB_MODEL`
  - (Optional) `ENVECTOR_USE_HF_DATASET=1`

3. Run the following command:
  
```bash
python -m pytest -q -m integration -s
```

## Contributing
See [`CONTRIBUTE.md`](CONTRIBUTE.md) for development, testing, and PR guidelines.
