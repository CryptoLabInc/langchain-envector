# Envector (LangChain VectorStore)

High-level VectorStore adaptor for Envector, using the `pyenvector` SDK. Vectors are encrypted on the server by default (`IndexSettings.index_encryption="cipher"`); the SDK performs the crypto client-side.

Key points
- Use high-level `pyenvector.EnvectorClient` and `pyenvector.Index`; avoid low-level `pyenvector.api.Indexer`/gRPC.
- Index encryption defaults to `cipher` (`IndexSettings.index_encryption`). Query can be `plain` or `cipher`.
- Metadata is stored as a single JSON string per item: `{text, metadata}`.
- Writes are asynchronous server-side. `WriteSettings` decides which wait: inserts return as soon as rows are searchable (no wait by default); update / upsert / delete wait by default so the change is visible to the next search.

Files
- `config.py`: Configuration dataclasses (connection, key, index).
- `client.py`: Initializes EnVector + index and returns an `Index` instance.
- `vectorstore.py`: `Envector` VectorStore implementation.
- `retriever.py`: Optional wrapper retriever.
- `examples/`: Minimal examples.

See `.cursor/general.md` for the full design and usage details.

