from __future__ import annotations

"""
05_llm_rag.py - LLM Semantic Retrieval + RAG
============================================
Pipeline:
  1. Build a text query from the user's interaction history.
  2. Embed the query with sentence-transformers (all-MiniLM-L6-v2).
  3. kNN search in Elasticsearch (pfg_<dataset>_items index) -> top candidates.
  4. Optionally re-rank candidates with a configurable backend (`ollama`, `gemini`, or `retrieval-only`).
  5. Return final top-10 recommendations and evaluate with the common metrics.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent))
from utils import (  # noqa: E402
    DATASETS,
    MODELS_DIR,
    add_common_args,
    apply_wandb_env,
    compute_metrics,
    encode_users_items,
    load_interactions,
    log,
    maybe_init_wandb,
    print_metrics,
    save_results,
    train_test_split_temporal,
)

MODEL_NAME = "llm_rag"
ES_HOST = "http://localhost:9200"
EMBED_MODEL = "all-MiniLM-L6-v2"
ES_CANDIDATES = 20
DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_MAX_USERS = 200


def _items_parquet(dataset: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "processed" / dataset / "items.parquet"


def build_item_text_map(dataset: str) -> dict[str, str]:
    import pandas as pd

    path = _items_parquet(dataset)
    if not path.exists():
        return {}

    df = pd.read_parquet(path)
    text_map: dict[str, str] = {}
    for _, row in df.iterrows():
        iid = str(row.get("item_id", ""))
        parts = []
        for col in [
            "title",
            "item_name",
            "artist_name",
            "track_name",
            "album_name",
            "name",
            "brand",
            "category",
            "categories",
            "city",
            "state",
            "country",
            "genres_str",
            "sample_summary",
            "summary_mode",
            "summary_examples",
            "feature_text",
            "description",
            "top_style",
            "temporal_profile",
        ]:
            val = row.get(col)
            if val and str(val) not in ("nan", "None", ""):
                parts.append(str(val))
        text_map[iid] = " | ".join(parts) if parts else iid
    return text_map


def build_user_history_text(user_item_ids: List[str], item_text_map: dict[str, str], top_n: int = 5) -> str:
    texts = [item_text_map.get(str(iid), str(iid)) for iid in user_item_ids[-top_n:]]
    return "; ".join(texts)


def knn_search(es, index: str, query_emb: np.ndarray, k: int = 20) -> List[tuple[str, str]]:
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
    return [
        (hit["_source"]["item_id"], hit["_source"].get("text_repr", ""))
        for hit in resp["hits"]["hits"]
    ]


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


def _extract_ranked_ids_from_content(content: str, k: int) -> List[str]:
    match = re.search(r"\[.*?\]", content, re.DOTALL)
    if not match:
        return []
    try:
        ranked_ids = json.loads(match.group())
    except Exception:
        return []
    return [str(r) for r in ranked_ids[:k]]


def _gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use Gemini as LLM reranker.")
    return api_key


def _gemini_rerank(prompt: str, model: str, timeout_seconds: float) -> str:
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={
            "x-goog-api-key": _gemini_api_key(),
            "Content-Type": "application/json",
        },
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
    return "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()


def _ollama_rerank(prompt: str, model: str, timeout_seconds: float) -> str:
    response = requests.post(
        os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "").strip()


def llm_rerank(
    history_text: str,
    candidates: List[tuple[str, str]],
    k: int = 10,
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str | None = None,
    timeout_seconds: float = 60.0,
) -> List[str]:
    provider_key = (provider or DEFAULT_LLM_PROVIDER).strip().lower()
    if provider_key in {"retrieval-only", "retrieval_only", "none"}:
        return [iid for iid, _ in candidates[:k]]

    cand_str = "\n".join(f"- {iid}: {txt}" for iid, txt in candidates)
    prompt = RANK_PROMPT.format(
        history=history_text,
        k=min(k, len(candidates)),
        candidates=cand_str,
    )

    try:
        if provider_key == "gemini":
            chosen_model = model or DEFAULT_GEMINI_MODEL
            content = _gemini_rerank(prompt, chosen_model, timeout_seconds)
        else:
            chosen_model = model or DEFAULT_OLLAMA_MODEL
            content = _ollama_rerank(prompt, chosen_model, timeout_seconds)

        ranked_ids = _extract_ranked_ids_from_content(content, k)
        if ranked_ids:
            return ranked_ids
    except Exception as exc:
        log.warning("%s rerank failed (%s), using retrieval order", provider, exc)

    return [iid for iid, _ in candidates[:k]]


def run(
    dataset: str,
    sample: int | None,
    llm_provider: str,
    llm_model: str,
    max_users: int,
    hparams: dict | None = None,
):
    log.info("\n%s\n[LLM+RAG] Dataset: %s\n%s", "=" * 60, dataset, "=" * 60)
    t_start = time.time()

    hp = hparams or {
        "es_host": ES_HOST,
        "embed_model": EMBED_MODEL,
        "es_candidates": ES_CANDIDATES,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "max_users": max_users,
    }

    try:
        from elasticsearch import Elasticsearch

        es = Elasticsearch(ES_HOST, request_timeout=30)
        es.info()
    except Exception as exc:
        log.error("Cannot connect to Elasticsearch: %s", exc)
        log.error("Run: docker compose up -d and python src/preprocessing/06_embed_and_index.py")
        return

    es_index = f"pfg_{dataset}_items"
    if not es.indices.exists(index=es_index):
        log.error("ES index '%s' not found. Run 06_embed_and_index.py first.", es_index)
        return

    from sentence_transformers import SentenceTransformer

    log.info("  Loading embedding model: %s ...", EMBED_MODEL)
    embed_model = SentenceTransformer(EMBED_MODEL)

    df = load_interactions(dataset, sample=sample)
    df, user_map, item_map = encode_users_items(df)
    n_items = len(item_map)
    train_df, test_df = train_test_split_temporal(df)

    wandb_run = maybe_init_wandb(
        model_name=MODEL_NAME,
        dataset_name=dataset,
        sample=sample,
        hparams={**hp, "n_items": n_items},
        run_name=f"{MODEL_NAME}-{dataset}-{llm_provider}-{llm_model}",
    )

    item_text_map = build_item_text_map(dataset)

    user_history = (
        train_df.sort_values("timestamp")
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )

    idx_to_uid = {v: k for k, v in user_map.items()}
    iid_to_idx = item_map
    item_emb_cache: dict[str, np.ndarray] = {}

    def get_item_embedding(item_id_str: str) -> Optional[np.ndarray]:
        if item_id_str in item_emb_cache:
            return item_emb_cache[item_id_str]
        try:
            resp = es.search(
                index=es_index,
                body={"query": {"term": {"item_id": item_id_str}}, "_source": ["embedding"], "size": 1},
            )
            hits = resp["hits"]["hits"]
            if hits:
                emb = np.array(hits[0]["_source"]["embedding"], dtype="float32")
                item_emb_cache[item_id_str] = emb
                return emb
        except Exception:
            pass
        return None

    emb_dim = 384
    item_embeddings = np.zeros((n_items, emb_dim), dtype="float32")

    def get_recs(user_idx: int, k: int = 10) -> List[int]:
        user_id = idx_to_uid.get(user_idx)
        if user_id is None:
            return []

        history_ids = user_history.get(str(user_id), [])
        history_text = build_user_history_text(history_ids, item_text_map)
        if not history_text.strip():
            history_text = str(user_id)

        query_emb = embed_model.encode(history_text, show_progress_bar=False)
        candidates = knn_search(es, es_index, query_emb, k=ES_CANDIDATES)
        if not candidates:
            return []

        for cand_id, _ in candidates:
            ci = iid_to_idx.get(str(cand_id))
            if ci is not None and item_embeddings[ci].sum() == 0:
                emb = get_item_embedding(str(cand_id))
                if emb is not None:
                    item_embeddings[ci] = emb

        reranked_ids = llm_rerank(
            history_text,
            candidates,
            k=k,
            provider=llm_provider,
            model=llm_model,
            timeout_seconds=float(os.getenv("RAG_LLM_TIMEOUT_SECONDS", "60")),
        )

        result: List[int] = []
        for iid in reranked_ids:
            idx = iid_to_idx.get(str(iid))
            if idx is not None:
                result.append(idx)
        return result[:k]

    log.info("  Evaluating with provider=%r model=%r ...", llm_provider, llm_model)

    def _progress(position: int, total: int, user_idx: int, recs: List[int]):
        if position == 1 or position == total or position % 5 == 0:
            log.info("  Progress: %s/%s users evaluated (user_idx=%s, n_recs=%s)", position, total, user_idx, len(recs))

    metrics = compute_metrics(
        get_recommendations=get_recs,
        test_df=test_df,
        item_embeddings=item_embeddings,
        catalog_size=n_items,
        max_users=int(hp["max_users"]),
        progress_callback=_progress,
    )
    metrics["train_time_s"] = 0
    metrics["llm_provider"] = llm_provider
    metrics["llm_model"] = llm_model
    print_metrics(metrics, MODEL_NAME, dataset)

    if wandb_run is not None:
        wandb_run.log({**metrics, "wall_time_s": round(time.time() - t_start, 1)})
        wandb_run.finish()

    out_dir = MODELS_DIR / MODEL_NAME / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    save_results(metrics, MODEL_NAME, dataset, hparams=hp, wandb_run=wandb_run)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="LLM + RAG Semantic Retrieval")
    add_common_args(parser)
    parser.add_argument(
        "--llm-provider",
        choices=["ollama", "gemini", "retrieval-only"],
        default=DEFAULT_LLM_PROVIDER,
        help="LLM backend used for candidate reranking.",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model name (default: {DEFAULT_GEMINI_MODEL})",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=DEFAULT_MAX_USERS,
        help=f"Maximum number of test users evaluated offline (default: {DEFAULT_MAX_USERS}).",
    )
    args = parser.parse_args()
    apply_wandb_env(args)
    sample = None if args.full else args.sample
    targets = [args.dataset] if args.dataset else DATASETS
    if args.llm_provider == "gemini":
        llm_model = args.gemini_model
    elif args.llm_provider == "retrieval-only":
        llm_model = "retrieval-only"
    else:
        llm_model = args.ollama_model
    for ds in targets:
        try:
            run(ds, sample, args.llm_provider, llm_model, args.max_users)
        except Exception as exc:
            log.error("[LLM+RAG] Failed on %s: %s", ds, exc, exc_info=True)


if __name__ == "__main__":
    main()
