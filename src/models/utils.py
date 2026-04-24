"""
Shared utilities for all model scripts.
"""
from __future__ import annotations

import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # pfg-models/
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR   = ROOT / "results"
MODELS_DIR    = ROOT / "weights"

DATASETS = ["movielens", "amazon_electronics", "yelp", "lastfm", "foursquare"]

# Implicit datasets use rating=1.0; explicit use original star ratings
IMPLICIT_DATASETS = {"lastfm", "foursquare"}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def add_wandb_args(parser):
    """
    Add Weights & Biases (wandb) CLI arguments.

    Logging is opt-in via --wandb (or PFG_WANDB=1 env var).
    """
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log runs to Weights & Biases (wandb).",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.getenv("WANDB_PROJECT", "pfg-recs"),
        help="wandb project name (default: pfg-recs).",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.getenv("WANDB_ENTITY", None),
        help="wandb entity/team (optional).",
    )
    parser.add_argument(
        "--wandb-group",
        default=os.getenv("WANDB_GROUP", None),
        help="wandb group (optional, useful for sweeps/benchmarks).",
    )
    parser.add_argument(
        "--wandb-tags",
        default=os.getenv("WANDB_TAGS", ""),
        help="Comma-separated wandb tags (optional).",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.getenv("WANDB_MODE", "online"),
        help="wandb mode: online/offline/disabled (default: online).",
    )
    return parser


def apply_wandb_env(args) -> None:
    """Apply wandb-related environment variables based on parsed CLI args."""
    if not getattr(args, "wandb", False):
        return

    os.environ["PFG_WANDB"] = "1"
    if getattr(args, "wandb_project", None):
        os.environ["WANDB_PROJECT"] = str(args.wandb_project)
    if getattr(args, "wandb_entity", None):
        os.environ["WANDB_ENTITY"] = str(args.wandb_entity)
    if getattr(args, "wandb_group", None):
        os.environ["WANDB_GROUP"] = str(args.wandb_group)
    tags_arg = getattr(args, "wandb_tags", None)
    if tags_arg is not None:
        tags_str = str(tags_arg).strip()
        if tags_str:
            os.environ["WANDB_TAGS"] = tags_str
        else:
            # Avoid wandb parsing empty tags as [""] and failing validation.
            os.environ.pop("WANDB_TAGS", None)
    if getattr(args, "wandb_mode", None):
        os.environ["WANDB_MODE"] = str(args.wandb_mode)


def wandb_enabled() -> bool:
    return os.getenv("PFG_WANDB", "").strip().lower() in {"1", "true", "yes", "y"}


def maybe_init_wandb(
    *,
    model_name: str,
    dataset_name: str,
    sample: int | None,
    hparams: dict | None = None,
    run_name: str | None = None,
):
    """Initialize a wandb run if enabled; otherwise returns None."""
    if not wandb_enabled():
        return None

    try:
        import wandb  # type: ignore
    except ImportError:
        log.warning("wandb is enabled but not installed. Install with: pip install wandb")
        return None

    project = os.getenv("WANDB_PROJECT", "pfg-recs")
    entity = os.getenv("WANDB_ENTITY") or None
    group = os.getenv("WANDB_GROUP") or None
    tags_raw = os.getenv("WANDB_TAGS", "")
    if not tags_raw.strip():
        os.environ.pop("WANDB_TAGS", None)
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] or None

    config = {
        "model": model_name,
        "dataset": dataset_name,
        "sample": sample,
    }
    if hparams:
        config.update(hparams)

    return wandb.init(
        project=project,
        entity=entity,
        group=group,
        tags=tags,
        name=run_name or f"{model_name}-{dataset_name}",
        config=config,
        reinit=True,
    )


# ── Data loading ──────────────────────────────────────────────────────────────
def load_interactions(
    dataset: str,
    sample: Optional[int] = 5_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load interactions.parquet for a dataset.
    If sample is not None and the dataset is larger, randomly sample
    up to `sample` rows (stratified by user to preserve user diversity).
    """
    path = PROCESSED_DIR / dataset / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")

    df = pd.read_parquet(path, columns=["user_id", "item_id", "rating", "timestamp"])
    log.info(f"[{dataset}] Loaded {len(df):,} interactions")

    if sample and len(df) > sample:
        # Keep all users with at least 1 interaction in sample (stratified)
        rng = np.random.default_rng(seed)
        sampled_users = rng.choice(
            df["user_id"].unique(),
            size=min(sample // 5, df["user_id"].nunique()),
            replace=False,
        )
        df = df[df["user_id"].isin(sampled_users)].reset_index(drop=True)
        if len(df) > sample:
            df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
        log.info(f"[{dataset}] Sampled to {len(df):,} interactions")

    # For implicit datasets, ensure rating=1.0
    if dataset in IMPLICIT_DATASETS:
        df["rating"] = 1.0

    return df


def train_test_split_temporal(
    df: pd.DataFrame,
    ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-last-out split: for each user, the most recent interaction(s)
    go to the test set. Prevents temporal data leakage.
    """
    df = df.sort_values(["user_id", "timestamp"])
    test_mask = df.groupby("user_id")["timestamp"].rank(ascending=False) <= max(1, int(ratio * 10))
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def encode_users_items(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
    """Map user_id and item_id to contiguous integer indices."""
    users = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
    items = {it: i for i, it in enumerate(sorted(df["item_id"].unique()))}
    df = df.copy()
    df["user_idx"] = df["user_id"].map(users).astype("int32")
    df["item_idx"] = df["item_id"].map(items).astype("int32")
    return df, users, items


def build_user_item_matrix(df: pd.DataFrame) -> csr_matrix:
    """Build a scipy sparse CSR matrix [n_users × n_items]."""
    n_users = df["user_idx"].max() + 1
    n_items = df["item_idx"].max() + 1
    mat = csr_matrix(
        (df["rating"].astype("float32"), (df["user_idx"], df["item_idx"])),
        shape=(n_users, n_items),
    )
    return mat


def build_sequences(
    df: pd.DataFrame,
    max_len: int = 50,
) -> Dict[int, List[int]]:
    """
    Return a dict {user_idx: [item_idx ordered by timestamp]}.
    Sequences are truncated to max_len (most recent items kept).
    """
    df = df.sort_values("timestamp")
    seqs: Dict[int, List[int]] = {}
    for user_idx, group in df.groupby("user_idx"):
        items = group["item_idx"].tolist()
        seqs[int(user_idx)] = items[-max_len:]
    return seqs


def negative_sample(
    df: pd.DataFrame,
    n_items: int,
    n_neg: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Add `n_neg` random negative item indices per row.
    Avoids items the user has interacted with.
    """
    rng = np.random.default_rng(seed)
    if n_neg < 1:
        raise ValueError("n_neg must be >= 1")

    user_items: Dict[int, set] = df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    # Expand rows so existing trainers can keep using a single neg_item_idx column.
    if n_neg == 1:
        expanded = df.copy()
        neg_items: list[int] = []
        for ui in expanded["user_idx"]:
            seen = user_items[int(ui)]
            while True:
                neg = int(rng.integers(0, n_items))
                if neg not in seen:
                    neg_items.append(neg)
                    break
        expanded["neg_item_idx"] = neg_items
        return expanded

    expanded = df.loc[df.index.repeat(n_neg)].reset_index(drop=True).copy()
    neg_items = []
    for ui in expanded["user_idx"]:
        seen = user_items[int(ui)]
        while True:
            neg = int(rng.integers(0, n_items))
            if neg not in seen:
                neg_items.append(neg)
                break
    expanded["neg_item_idx"] = neg_items
    return expanded


# ── Metrics ───────────────────────────────────────────────────────────────────
def _dcg_at_k(relevances: List[int], k: int) -> float:
    relevances = relevances[:k]
    gains = [rel / np.log2(i + 2) for i, rel in enumerate(relevances)]
    return sum(gains)


def ndcg_at_k(recommended: List[int], relevant: set, k: int = 10) -> float:
    hits = [1 if item in relevant else 0 for item in recommended[:k]]
    ideal = sorted(hits, reverse=True)
    dcg  = _dcg_at_k(hits, k)
    idcg = _dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(recommended: List[int], relevant: set, k: int = 10) -> float:
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant) if relevant else 0.0


def ild_at_k(recommended: List[int], item_embeddings: np.ndarray, k: int = 10) -> float:
    """Intra-List Diversity: mean pairwise cosine distance in top-k."""
    recs = recommended[:k]
    if len(recs) < 2:
        return 0.0
    embs = item_embeddings[recs]                     # [k, dim]
    # Normalise
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    embs = embs / norms
    sim_matrix = embs @ embs.T                       # [k, k]
    k_actual = len(recs)
    # Average upper triangle (exclude diagonal)
    total = 0.0
    count = 0
    for i in range(k_actual):
        for j in range(i + 1, k_actual):
            total += 1 - sim_matrix[i, j]            # distance = 1 - cosine_sim
            count += 1
    return total / count if count > 0 else 0.0


def compute_metrics(
    get_recommendations,          # callable(user_idx) -> List[int] top-k items
    test_df: pd.DataFrame,
    item_embeddings: np.ndarray,
    catalog_size: int,
    k: int = 10,
    max_users: int = 2000,
) -> dict:
    """
    Evaluate a recommendation function over test users.
    Records latency per query.

    Returns: {ndcg, recall, latency_p50, latency_p95, ild, coverage}
    """
    ndcgs, recalls, ilds, latencies = [], [], [], []
    all_recommended: set = set()

    test_users = test_df["user_idx"].unique()
    if len(test_users) > max_users:
        rng = np.random.default_rng(0)
        test_users = rng.choice(test_users, size=max_users, replace=False)

    for user_idx in test_users:
        relevant = set(test_df[test_df["user_idx"] == user_idx]["item_idx"].tolist())
        if not relevant:
            continue

        t0 = time.perf_counter()
        recs = get_recommendations(int(user_idx))
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

        ndcgs.append(ndcg_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ilds.append(ild_at_k(recs, item_embeddings, k))
        all_recommended.update(recs[:k])

    latencies_sorted = sorted(latencies)
    p50_idx = int(0.50 * len(latencies_sorted))
    p95_idx = int(0.95 * len(latencies_sorted))

    return {
        "ndcg@10":        round(float(np.mean(ndcgs)), 4),
        "recall@10":      round(float(np.mean(recalls)), 4),
        "latency_p50_ms": round(latencies_sorted[p50_idx] if latencies_sorted else 0, 2),
        "latency_p95_ms": round(latencies_sorted[p95_idx] if latencies_sorted else 0, 2),
        "ild@10":         round(float(np.mean(ilds)), 4),
        "coverage@10":    round(len(all_recommended) / catalog_size, 4),
    }


# ── Saving results ─────────────────────────────────────────────────────────────
def save_results(
    metrics: dict,
    model_name: str,
    dataset_name: str,
    *,
    hparams: dict | None = None,
    wandb_run=None,
) -> Path:
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}.json"
    payload = {"model": model_name, "dataset": dataset_name, **metrics}
    if hparams:
        payload["hparams"] = hparams
    if wandb_run is not None:
        run_id = getattr(wandb_run, "id", None)
        run_name = getattr(wandb_run, "name", None)
        if run_id or run_name:
            payload["wandb"] = {"id": run_id, "name": run_name}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Results saved → {path}")
    return path


def print_metrics(metrics: dict, model_name: str, dataset_name: str):
    log.info(f"\n{'─'*60}")
    log.info(f"  {model_name}  ×  {dataset_name}")
    log.info(f"  NDCG@10       : {metrics['ndcg@10']}")
    log.info(f"  Recall@10     : {metrics['recall@10']}")
    log.info(f"  Latency p50   : {metrics['latency_p50_ms']} ms")
    log.info(f"  Latency p95   : {metrics['latency_p95_ms']} ms")
    log.info(f"  ILD@10        : {metrics['ild@10']}")
    log.info(f"  Coverage@10   : {metrics['coverage@10']}")
    log.info(f"{'─'*60}")


def add_common_args(parser):
    """Add standard CLI arguments to any model's argparse."""
    parser.add_argument(
        "--dataset", choices=DATASETS, default=None,
        help="Run on a single dataset. Omit to run on all.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Disable sampling (use full dataset — slow on CPU).",
    )
    parser.add_argument(
        "--sample", type=int, default=5_000_000,
        help="Max interactions to sample per dataset (default 5M).",
    )
    add_wandb_args(parser)
    return parser
