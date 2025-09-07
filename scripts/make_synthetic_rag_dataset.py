"""Generate a synthetic RAG dataset (~1000 passages) without external deps.

Outputs JSONL at `data/synthetic_rag_1k.jsonl` with fields:
  {"text": str, "metadata": {"topic": str, "idx": int}}

Usage (from repo root):
  python scripts/make_synthetic_rag_dataset.py [--size 1000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


TOPICS = [
    "cryptography",
    "machine learning",
    "distributed systems",
    "databases",
    "data engineering",
    "system design",
    "networking",
    "operating systems",
    "compilers",
    "software testing",
    "security",
    "cloud computing",
    "devops",
    "productivity",
    "nlp",
    "computer vision",
    "recommendation",
    "reinforcement learning",
    "mathematics",
    "statistics",
]

TEMPLATES = [
    "An introduction to {topic} with practical examples and common pitfalls.",
    "Key concepts in {topic} explained for engineers and researchers.",
    "Hands-on guide: building small projects using {topic} in production.",
    "Frequently asked questions about {topic} and concise answers.",
    "A short overview of tools and libraries used in {topic} today.",
    "Performance considerations and benchmarking approaches within {topic}.",
    "Design patterns that work well when applying {topic} at scale.",
    "Migration checklist for teams adopting {topic} into existing systems.",
    "Glossary of terms commonly used by practitioners of {topic}.",
    "Case studies summarizing lessons learned while deploying {topic}.",
]


def make_sentence(topic: str) -> str:
    return random.choice(TEMPLATES).format(topic=topic)


def make_paragraph(topic: str, min_sent: int = 3, max_sent: int = 7) -> str:
    n = random.randint(min_sent, max_sent)
    return " " .join(make_sentence(topic) for _ in range(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_rag_1k.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.size):
            topic = random.choice(TOPICS)
            text = make_paragraph(topic)
            rec = {"text": text, "metadata": {"topic": topic, "idx": i}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {args.size} records to {out_path}")


if __name__ == "__main__":
    main()

