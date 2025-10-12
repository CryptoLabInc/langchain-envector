"""Export a public Hugging Face dataset to JSONL for RAG ingestion.

Examples:
  # Take first 1000 rows of ag_news train split
  python scripts/export_hf_dataset.py --name ag_news --split train --text-column text --size 1000

  # Sample 1000 rows at random and keep a label as metadata
  python scripts/export_hf_dataset.py \
    --name ag_news --split train --text-column text --meta-columns label --size 1000 --seed 42

Outputs JSONL at data/hf_export.jsonl with {"text": str, "metadata": {...}} per line.
Requires: datasets (pip install datasets)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="HF dataset name, e.g., ag_news")
    ap.add_argument(
        "--subset", default=None, help="Optional subset/config of the dataset"
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-column", required=True)
    ap.add_argument(
        "--meta-columns",
        nargs="*",
        default=[],
        help="Optional metadata columns to carry over",
    )
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/hf_export.jsonl")
    args = ap.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise SystemExit(f"Install 'datasets' package to use this script: {e}")

    ds = load_dataset(args.name, args.subset, split=args.split)
    if args.size and args.size < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.size))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row[args.text_column]
            meta = (
                {k: row.get(k) for k in args.meta_columns} if args.meta_columns else {}
            )
            rec = {"text": text, "metadata": meta}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds)} rows to {out_path}")


if __name__ == "__main__":
    main()
