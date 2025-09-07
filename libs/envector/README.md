# Envector (LangChain VectorStore)

High-level VectorStore adaptor for Envector (ES2), using the `es2` SDK. Vectors are always encrypted on the server; the SDK performs required crypto client-side.

Key points
- Use high-level `es2.ES2` and `es2.Index`; avoid low-level `es2.api.Indexer`/gRPC.
- Index encryption is fixed to `cipher`. Query can be `plain` or `cipher`.
- Metadata is stored as a single JSON string per item: `{id, text, metadata}`.

Files
- `config.py`: Configuration dataclasses (connection, key, index).
- `client.py`: Initializes ES2 + index and returns an `Index` instance.
- `vectorstore.py`: `Envector` VectorStore implementation.
- `retriever.py`: Optional wrapper retriever.
- `examples/`: Minimal examples.

See `.cursor/general.md` for the full design and usage details.

