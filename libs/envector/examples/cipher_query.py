"""Encrypted query example.

Demonstrates running a search where the query is encrypted (`query_encryption="cipher"`).
Server-side vectors remain encrypted as always.
"""

from __future__ import annotations

from libs.envector.config import (
    ConnectionConfig,
    EnvectorConfig,
    IndexSettings,
    KeyConfig,
)
from libs.envector.vectorstore import Envector


def main():
    cfg = EnvectorConfig(
        connection=ConnectionConfig(address="localhost:50050"),
        key=KeyConfig(
            key_path="./keys", key_id="example_key", preset="ip", eval_mode="rmp"
        ),
        index=IndexSettings(
            index_name="demo_cipher", dim=384, query_encryption="cipher"
        ),
        create_if_missing=True,
    )

    # Embeddings are still required to produce the query vector.
    from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = Envector(config=cfg, embeddings=emb)

    # Add some small texts (inserts will encrypt vectors automatically)
    store.add_texts(
        texts=["Encrypted vector search is privacy-preserving"],
        metadatas=[{"topic": "encryption"}],
    )

    # Search with encrypted query
    docs = store.similarity_search("privacy-preserving search", k=1)
    for d in docs:
        print("DOC:", d.page_content, d.metadata)


if __name__ == "__main__":
    main()
