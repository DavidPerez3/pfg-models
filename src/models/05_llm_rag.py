"""
05_llm_rag.py — LLM Semantic Retrieval + RAG via Ollama
=========================================================
Pipeline:
  1. Build a text query from the user's interaction history.
  2. Embed the query with sentence-transformers (all-MiniLM-L6-v2).
  3. kNN search in Elasticsearch (pfg_<dataset>_items index) → top-20 candidates.
  4. Re-rank candidates with an Ollama LLM (no API key, fully local).
  5. Return final top-10 recommendations.

Prerequisites:
  - Elasticsearch running: docker compose up -d
  - Items indexed: python src/preprocessing/06_embed_and_index.py
  - Ollama running with a model pulled, e.g.: ollama pull llama3.2

Usage:
    python src/models/05_llm_rag.py --dataset lastfm --ollama-model llama3.2
    python src/models/05_llm_rag.py   # all datasets
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DATASETS, MODELS_DIR,
    add_common_args, compute_metrics, encode_users_items,
    load_interactions, log, print_metrics, save_results,
    train_test_split_temporal,
    apply_wandb_env, maybe_init_wandb,
)

MODEL_NAME   = "llm_rag"
ES_HOST      = "http://localhost:9200"
EMBED_MODEL  = "all-MiniLM-L6-v2"
ES_CANDIDATES = 20
DEFAULT_OLLAMA_MODEL = "llama3.2"

# ── Text representation builders (per dataset) ───────────────────────────────
def _items_parquet(dataset: str):
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    p = ROOT / "data" / "processed" / dataset / "items.parquet"
    return p


def build_item_text_map(dataset: str) -> dict:
    """
    Returns {item_id (str): text_repr (str)} for all items in items.parquet.
    Gracefully handles missing columns.
    """
    import pandas as pd
    path = _items_parquet(dataset)
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    text_map = {}
    for _, row in df.iterrows():
        iid = str(row.get("item_id", ""))
        parts = []
        for col in ["title", "artist_name", "track_name", "album_name",
                    "name", "categories", "city", "genres_str"]:
            val = row.get(col)
            if val and str(val) not in ("nan", "None", ""):
                parts.append(str(val))
        text_map[iid] = " | ".join(parts) if parts else iid
    return text_map


def build_user_history_text(user_item_ids: List[str], item_text_map: dict, top_n: int = 5) -> str:
    texts = [item_text_map.get(str(iid), str(iid)) for iid in user_item_ids[-top_n:]]
    return "; ".join(texts)


# ── Elasticsearch kNN search ─────────────────────────────────────────────────
def knn_search(es, index: str, query_emb: np.ndarray, k: int = 20) -> List[str]:
    """Return top-k item_id strings from ES kNN search."""
    resp = es.search(
        index=index,
        body={
            "knn": {
                "field": "embedding",
                "query_vector": query_emb.tolist(),
                "k": k,
                "num_candidates": k * 5,
            },
            "_source": ["item_id", "text_repr"],
            "size": k,
        },
    )
    return [(hit["_source"]["item_id"], hit["_source"].get("text_repr", ""))
            for hit in resp["hits"]["hits"]]


# ── Ollama re-ranking ─────────────────────────────────────────────────────────
RANK_PROMPT = """\
You are a recommendation system assistant.

A user has enjoyed the following items (most recent first):
{history}

From the candidates below, select the {k} most relevant items to recommend next.
Return ONLY a JSON array of the item IDs in ranked order, e.g.:
["id1", "id2", "id3"]

Candidates:
{candidates}
"""

def ollama_rerank(
    history_text: str,
    candidates: List[tuple],   # [(item_id, text_repr), ...]
    k: int = 10,
    model: str = DEFAULT_OLLAMA_MODEL,
) -> List[str]:
    """
    Call Ollama to re-rank candidate items.
    Falls back to original ES order on any error.
    """
    try:
        import ollama
        cand_str = "\n".join(f"- {iid}: {txt}" for iid, txt in candidates)
        prompt = RANK_PROMPT.format(
            history=history_text,
            k=min(k, len(candidates)),
            candidates=cand_str,
        )
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        content = response["message"]["content"].strip()
        # Extract JSON array from response
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            ranked_ids = json.loads(match.group())
            return [str(r) for r in ranked_ids[:k]]
    except Exception as e:
        log.warning(f"Ollama re-rank failed ({e}), using ES order")
    return [iid for iid, _ in candidates[:k]]


# ── Main run ──────────────────────────────────────────────────────────────────
def run(
    dataset: str,
    sample: int | None,
    ollama_model: str,
    hparams: dict | None = None,
):
    log.info(f"\n{'='*60}\n[LLM+RAG] Dataset: {dataset}\n{'='*60}")
    t_start = time.time()

    hp = hparams or {
        "es_host": ES_HOST,
        "embed_model": EMBED_MODEL,
        "es_candidates": ES_CANDIDATES,
        "ollama_model": ollama_model,
        "max_users": 200,
    }

    # Check ES connectivity
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(ES_HOST, request_timeout=30)
        es.info()
    except Exception as e:
        log.error(f"Cannot connect to Elasticsearch: {e}")
        log.error("Run: docker compose up -d  and  python src/preprocessing/06_embed_and_index.py")
        return

    # Check ES index exists
    es_index = f"pfg_{dataset}_items"
    if not es.indices.exists(index=es_index):
        log.error(f"ES index '{es_index}' not found. Run 06_embed_and_index.py first.")
        return

    # Load sentence-transformer
    from sentence_transformers import SentenceTransformer
    log.info(f"  Loading embedding model: {EMBED_MODEL} …")
    embed_model = SentenceTransformer(EMBED_MODEL)

    # Load interactions
    df = load_interactions(dataset, sample=sample)
    df, user_map, item_map = encode_users_items(df)
    n_items = len(item_map)
    train_df, test_df = train_test_split_temporal(df)

    wandb_run = maybe_init_wandb(
        model_name=MODEL_NAME,
        dataset_name=dataset,
        sample=sample,
        hparams={**hp, "n_items": n_items},
        run_name=f"{MODEL_NAME}-{dataset}-{ollama_model}",
    )

    # Build item text map
    item_text_map = build_item_text_map(dataset)

    # Build user history (item_id strings, not idx)
    user_history = (
        train_df.sort_values("timestamp")
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )

    # Build reverse mapping idx → item_id string, and item_id → idx
    idx_to_iid = {v: k for k, v in item_map.items()}
    iid_to_idx = item_map

    # Dummy item embeddings for ILD (use ES-based representations)
    # We'll fetch from ES lazily and cache
    item_emb_cache = {}

    def get_item_embedding(item_id_str: str) -> Optional[np.ndarray]:
        if item_id_str in item_emb_cache:
            return item_emb_cache[item_id_str]
        try:
            resp = es.search(
                index=es_index,
                body={"query": {"term": {"item_id": item_id_str}},
                      "_source": ["embedding"], "size": 1},
            )
            hits = resp["hits"]["hits"]
            if hits:
                emb = np.array(hits[0]["_source"]["embedding"], dtype="float32")
                item_emb_cache[item_id_str] = emb
                return emb
        except Exception:
            pass
        return None

    # Build a dense embedding matrix [n_items, dim] for ILD computation
    # (will be computed lazily during eval; use zeros as fallback)
    EMB_DIM = 384
    item_embeddings = np.zeros((n_items, EMB_DIM), dtype="float32")

    def get_recs(user_idx: int, k: int = 10) -> List[int]:
        user_id = idx_to_iid.get(user_idx)
        if user_id is None:
            return []
        history_ids = user_history.get(str(user_id), [])
        history_text = build_user_history_text(history_ids, item_text_map)
        if not history_text.strip():
            history_text = str(user_id)

        # Embed query
        query_emb = embed_model.encode(history_text, show_progress_bar=False)

        # kNN retrieve
        candidates = knn_search(es, es_index, query_emb, k=ES_CANDIDATES)
        if not candidates:
            return []

        # Populate item_embeddings for ILD
        for cand_id, _ in candidates:
            ci = iid_to_idx.get(str(cand_id))
            if ci is not None and item_embeddings[ci].sum() == 0:
                emb = get_item_embedding(str(cand_id))
                if emb is not None:
                    item_embeddings[ci] = emb

        # Ollama re-rank
        reranked_ids = ollama_rerank(history_text, candidates, k=k, model=ollama_model)

        # Convert item_id strings → item_idx integers
        result = []
        for iid in reranked_ids:
            idx = iid_to_idx.get(str(iid))
            if idx is not None:
                result.append(idx)
        return result[:k]

    log.info(f"  Evaluating with Ollama model: {ollama_model} …")
    metrics = compute_metrics(
        get_recommendations=get_recs,
        test_df=test_df,
        item_embeddings=item_embeddings,
        catalog_size=n_items,
        max_users=int(hp["max_users"]),   # LLM inference is slow — limit eval users
    )
    metrics["train_time_s"] = 0  # no training
    metrics["ollama_model"] = ollama_model
    print_metrics(metrics, MODEL_NAME, dataset)

    if wandb_run is not None:
        wandb_run.log({**metrics, "wall_time_s": round(time.time() - t_start, 1)})
        wandb_run.finish()

    out_dir = MODELS_DIR / MODEL_NAME / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    save_results(metrics, MODEL_NAME, dataset, hparams=hp, wandb_run=wandb_run)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="LLM + RAG Semantic Retrieval (Ollama)")
    add_common_args(parser)
    parser.add_argument(
        "--ollama-model", default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL})",
    )
    args = parser.parse_args()
    apply_wandb_env(args)
    sample = None if args.full else args.sample
    targets = [args.dataset] if args.dataset else DATASETS
    for ds in targets:
        try:
            run(ds, sample, args.ollama_model)
        except Exception as e:
            log.error(f"[LLM+RAG] Failed on {ds}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
