"""Basic usage example for Envector VectorStore.

Requirements:
- `pyenvector`
- `langchain` (version providing VectorStore APIs)
- An embeddings backend, e.g. sentence-transformers
"""

from __future__ import annotations

from langchain_envector.config import (
    ConnectionConfig,
    EnvectorConfig,
    IndexSettings,
    KeyConfig,
)
from langchain_envector.vectorstore import Envector


def main():
    # Replace with your actual settings
    cfg = EnvectorConfig(
        connection=ConnectionConfig(address="localhost:50050"),
        key=KeyConfig(
            key_path="./keys", key_id="example_key", preset="ip2", eval_mode="mm32"
        ),
        index=IndexSettings(index_name="demo", dim=384, query_encryption="plain"),
        create_if_missing=True,
    )

    # Lazy import embeddings to keep example self-contained
    from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = Envector(config=cfg, embeddings=emb)

    ids = store.add_texts(
        texts=[
            "Homomorphic encryption enables computation over ciphertext",
            "LangChain integrates retrieval with LLMs",
        ],
        metadatas=[{"source": "note1"}, {"source": "note2"}],
    )
    print("inserted ids:", ids)

    docs = store.similarity_search("What computes on encrypted data?", k=2)
    for d in docs:
        print("DOC:", d.page_content, d.metadata)


if __name__ == "__main__":
    main()
