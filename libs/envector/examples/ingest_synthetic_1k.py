"""Ingest the synthetic 1K dataset into Envector.

Requires:
- EnVector server and keys.
- Dataset at `data/synthetic_rag_1k.jsonl` (run scripts/make_synthetic_rag_dataset.py).

Usage:
  python libs/envector/examples/ingest_synthetic_1k.py \
    --address 0.0.0.0:51050 \
    --key-path ./keys --key-id example_key \
    --index-name rag_1k [--dim 384] \
    [--use-embeddings] [--model sentence-transformers/all-MiniLM-L6-v2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from langchain_envector.config import (
    ConnectionConfig,
    EnvectorConfig,
    IndexSettings,
    KeyConfig,
)
from langchain_envector.vectorstore import Envector


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", required=True)
    ap.add_argument("--key-path", required=True)
    ap.add_argument("--key-id", required=True)
    ap.add_argument("--index-name", required=True)
    ap.add_argument(
        "--dim",
        type=int,
        required=False,
        help="If omitted and --use-embeddings, infer from model.",
    )
    ap.add_argument("--dataset", default="data/synthetic_rag_1k.jsonl")
    ap.add_argument("--use-embeddings", action="store_true")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    inferred_dim = None
    if args.use_embeddings:
        from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore

        embeddings = HuggingFaceEmbeddings(model_name=args.model)
        inferred_dim = len(embeddings.embed_query("_probe_"))
    else:
        embeddings = None

    cfg = EnvectorConfig(
        connection=ConnectionConfig(address=args.address),
        key=KeyConfig(
            key_path=args.key_path, key_id=args.key_id, preset="ip2", eval_mode="mm32"
        ),
        index=IndexSettings(
            index_name=args.index_name,
            dim=(args.dim if args.dim is not None else inferred_dim or 0),
            query_encryption="plain",
        ),
        create_if_missing=True,
    )
    store = Envector(config=cfg, embeddings=embeddings)

    path = Path(args.dataset)
    texts: List[str] = []
    metas: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])  # type: ignore
            metas.append(rec.get("metadata", {}))

    # Insert in batches to respect SDK batch behavior
    for t_batch, m_batch in zip(batched(texts, args.batch), batched(metas, args.batch)):
        if embeddings is None:
            # Without embeddings, require manual vectors; here we simply skip.
            # Users should provide --use-embeddings or adapt to their vector source.
            raise ValueError(
                "--use-embeddings is required unless you provide vectors explicitly."
            )
        store.add_texts(t_batch, metadatas=m_batch)

    print(f"Inserted {len(texts)} documents into index '{args.index_name}'")


if __name__ == "__main__":
    main()
