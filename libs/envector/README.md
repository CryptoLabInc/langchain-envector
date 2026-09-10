# Envector (LangChain VectorStore)

High-level VectorStore adaptor for Envector, using the `pyenvector` SDK. Vectors are always encrypted on the server; the SDK performs required crypto client-side.

Key points
- Use high-level `pyenvector.EnvectorClient` and `pyenvector.Index`; avoid low-level `pyenvector.api.Indexer`/gRPC.
- Index encryption is fixed to `cipher`. Query can be `plain` or `cipher`.
- Metadata is stored as a single JSON string per item: `{text, metadata}`.
- Writes (`insert` / `delete` / `update` / `upsert`) are asynchronous server-side; `WriteSettings` makes them wait for completion by default so a write is visible to the next search.

Files
- `config.py`: Configuration dataclasses (connection, key, index).
- `client.py`: Initializes EnVector + index and returns an `Index` instance.
- `vectorstore.py`: `Envector` VectorStore implementation.
- `retriever.py`: Optional wrapper retriever.
- `examples/`: Minimal examples.

See `.cursor/general.md` for the full design and usage details.

