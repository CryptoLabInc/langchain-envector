from __future__ import annotations

import os
import secrets
import time
import pytest

from langchain_envector.config import (
    ConnectionConfig,
    EnvectorConfig,
    IndexSettings,
    KeyConfig,
)
from langchain_envector.vectorstore import Envector


pytestmark = pytest.mark.integration


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        pytest.skip(f"Set {name} to enable integration test")
    return v


@pytest.mark.skipif(
    os.environ.get("ES2_ADDRESS") is None,
    reason="Set ES2_ADDRESS (e.g., 0.0.0.0:50050) to enable ES2 integration tests",
)
def test_e2e_vectorstore_plain_and_cipher():
    try:
        import es2  # type: ignore
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"es2 SDK not available: {e}")

    address = _require_env("ES2_ADDRESS")
    key_path = _require_env("ES2_KEY_PATH")
    key_id = _require_env("ES2_KEY_ID")
    use_emb = os.environ.get("ES2_USE_EMBEDDINGS") in {"1", "true", "TRUE", "yes"}
    model_name = os.environ.get(
        "ES2_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    use_hf = os.environ.get("ES2_USE_HF_DATASET") in {"1", "true", "TRUE", "yes"}
    hf_name = os.environ.get("ES2_HF_NAME", "ag_news")
    hf_subset = os.environ.get("ES2_HF_SUBSET")
    hf_split = os.environ.get("ES2_HF_SPLIT", "train")
    hf_text_col = os.environ.get("ES2_HF_TEXT_COL", "text")
    hf_meta_cols = [
        c for c in os.environ.get("ES2_HF_META_COLS", "label").split(",") if c
    ]
    hf_size = int(os.environ.get("ES2_HF_SIZE", "200"))
    hf_seed = int(os.environ.get("ES2_HF_SEED", "42"))

    # Determine dimension: either from env, or from embeddings model, or default
    dim_env = os.environ.get("ES2_DIM")
    if use_emb:
        emb = None
        # Prefer LangChain embeddings if available, else fall back to sentence-transformers
        try:
            from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore

            emb = HuggingFaceEmbeddings(model_name=model_name)
            dim = len(emb.embed_query("test"))
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                st = SentenceTransformer(model_name)
                # as_embeddings() can adapt this
                emb = st  # type: ignore[assignment]
                dim = len(st.encode(["test"])[0])
            except Exception as e:
                pytest.skip(f"Embeddings requested but unavailable: {e}")
    else:
        dim = int(dim_env or "16")

    if dim < 16 or dim > 4096:
        pytest.skip("Envector supports dimensions in [16, 4096]")

    base_index_name = os.environ.get(
        "ES2_INDEX_NAME", f"inttest_{secrets.token_hex(4)}"
    )

    import es2

    es2.init_connect(address=address)
    es2.reset()

    # Plain query mode
    cfg_plain = EnvectorConfig(
        connection=ConnectionConfig(address=address),
        key=KeyConfig(key_path=key_path, key_id=key_id, preset="ip", eval_mode="rmp"),
        index=IndexSettings(
            index_name=f"{base_index_name}_plain", dim=dim, query_encryption="plain"
        ),
        create_if_missing=True,
    )
    store_plain = Envector(config=cfg_plain, embeddings=(emb if use_emb else None))

    if use_hf:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as e:
            pytest.skip(f"HF datasets requested but unavailable: {e}")
        ds = load_dataset(hf_name, hf_subset, split=hf_split)
        if hf_size and hf_size < len(ds):
            ds = ds.shuffle(seed=hf_seed).select(range(hf_size))
        texts = [row[hf_text_col] for row in ds]
        metas = [{k: row.get(k) for k in hf_meta_cols if k in row} for row in ds]
        print(texts[0])
        print(metas[0])
    else:
        texts = [
            "machine learning accelerates research",
            "cooking recipes are delicious",
        ]
        metas = [{"label": "A"}, {"label": "B"}]

    if use_emb:
        store_plain.add_texts(texts, metadatas=metas)
    else:
        # Insert two simple orthogonal-ish vectors (limit to first two)
        e1 = [1.0] + [0.0] * (dim - 1)
        e2 = [0.0, 1.0] + [0.0] * (dim - 2)
        store_plain.add_texts(texts[:2], metadatas=metas[:2], vectors=[e1, e2])

    # Give server a moment if needed
    time.sleep(0.2)

    # Search
    if use_emb:
        q1 = "machine learning" if not use_hf else texts[0].split(" ")[0]
        docs = store_plain.similarity_search(q1, k=3)
        print("[plain] top-3 results for:", q1)
        for d in docs:
            print(
                " - score=",
                d.metadata.get("_score"),
                "text=",
                (d.page_content[:80] + ("..." if len(d.page_content) > 80 else "")),
            )
        assert len(docs) >= 1
        assert all("_id" in d.metadata for d in docs)
        # optional filter check if 'label' is part of meta
        if not use_hf:
            docs_f = store_plain.similarity_search(
                "cooking", k=2, filter={"label": "B"}
            )
            print("[plain] filtered results (label=B):", [d.metadata for d in docs_f])
            assert len(docs_f) >= 1 and all(
                d.metadata.get("label") == "B" for d in docs_f
            )
    else:
        # Using explicit embeddings
        docs = store_plain.similarity_search("q", k=2, embedding=e1)
        print(
            "[plain] results (explicit embedding e1):", [d.page_content for d in docs]
        )
        assert any(d.page_content == texts[0] for d in docs)
        assert all("_id" in d.metadata for d in docs)
        docs_f = store_plain.similarity_search(
            "q", k=2, embedding=e2, filter={"label": "B"}
        )
        print("[plain] filtered (e2, label=B):", [d.page_content for d in docs_f])
        assert len(docs_f) >= 1
        assert docs_f[0].page_content == texts[1]

    # Cipher query mode
    cfg_cc = EnvectorConfig(
        connection=ConnectionConfig(address=address),
        key=KeyConfig(key_path=key_path, key_id=key_id, preset="ip", eval_mode="rmp"),
        index=IndexSettings(
            index_name=f"{base_index_name}_cipher", dim=dim, query_encryption="cipher"
        ),
        create_if_missing=True,
    )
    store_cc = Envector(config=cfg_cc, embeddings=(emb if use_emb else None))
    if use_emb:
        store_cc.add_texts(texts, metadatas=metas)
    else:
        store_cc.add_texts(texts[:2], metadatas=metas[:2], vectors=[e1, e2])

    time.sleep(0.2)
    if use_emb:
        q2 = "cooking" if not use_hf else texts[-1].split(" ")[0]
        docs_cc = store_cc.similarity_search(q2, k=3)
        print("[cipher] top-3 results for:", q2)
        for d in docs_cc:
            print(
                " - score=",
                d.metadata.get("_score"),
                "text=",
                (d.page_content[:80] + ("..." if len(d.page_content) > 80 else "")),
            )
        assert len(docs_cc) >= 1
        assert all("_id" in d.metadata for d in docs_cc)
    else:
        docs_cc = store_cc.similarity_search("q", k=2, embedding=e2)
        print(
            "[cipher] results (explicit embedding e2):",
            [d.page_content for d in docs_cc],
        )
        assert any(d.page_content == texts[1] for d in docs_cc)
        assert all("_id" in d.metadata for d in docs_cc)

    # Cleanup
    store_plain.client.es2.drop_index(cfg_plain.index.index_name)
    store_cc.client.es2.drop_index(cfg_cc.index.index_name)
