"""
Shared utility functions for all preprocessing scripts.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]        # pfg-models/
DATA_DIR = ROOT / "data" / "raw"                  # raw source files
PROCESSED_DIR = ROOT / "data" / "processed"        # output Parquet files

REQUIRED_COLS = ["user_id", "item_id", "rating", "timestamp", "dataset"]

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Core helpers ─────────────────────────────────────────────────────────────
def validate_schema(df: pd.DataFrame, name: str) -> None:
    """Assert all mandatory columns are present and have zero nulls."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing mandatory columns: {missing}")
    null_counts = df[REQUIRED_COLS].isnull().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise ValueError(f"[{name}] Null values in mandatory columns:\n{bad}")
    assert df["rating"].between(0, 5).all(), f"[{name}] rating values out of [0,5] range"
    assert (df["timestamp"] > 0).all(), f"[{name}] Non-positive timestamps detected"


def report_stats(df: pd.DataFrame, name: str) -> None:
    """Print a concise stats summary to stdout."""
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_rows = len(df)
    sparsity = 1 - n_rows / (n_users * n_items) if n_users * n_items > 0 else float("nan")
    log.info("=" * 60)
    log.info(f"  Dataset      : {name}")
    log.info(f"  Rows         : {n_rows:,}")
    log.info(f"  Users        : {n_users:,}")
    log.info(f"  Items        : {n_items:,}")
    log.info(f"  Sparsity     : {sparsity:.6f}")
    log.info(f"  Rating range : [{df['rating'].min():.2f}, {df['rating'].max():.2f}]")
    log.info(f"  Rating mean  : {df['rating'].mean():.3f}")
    log.info("=" * 60)


def save_processed(df: pd.DataFrame, name: str, extra: Optional[pd.DataFrame] = None) -> Path:
    """
    Save interactions DataFrame (and optional item metadata) to Parquet.
    Returns the output directory.
    """
    out_dir = PROCESSED_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    interactions_path = out_dir / "interactions.parquet"
    df.to_parquet(interactions_path, index=False, engine="pyarrow")
    log.info(f"Saved interactions → {interactions_path}")

    if extra is not None:
        items_path = out_dir / "items.parquet"
        extra.to_parquet(items_path, index=False, engine="pyarrow")
        log.info(f"Saved item metadata → {items_path}")

    return out_dir


def kcore_filter(df: pd.DataFrame, k: int = 5, user_col: str = "user_id", item_col: str = "item_id") -> pd.DataFrame:
    """
    Iteratively remove users and items that have fewer than k interactions.
    Converges when no more rows are removed.
    """
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)
        user_counts = df[user_col].value_counts()
        df = df[df[user_col].isin(user_counts[user_counts >= k].index)]
        item_counts = df[item_col].value_counts()
        df = df[df[item_col].isin(item_counts[item_counts >= k].index)]
    log.info(f"After {k}-core filtering: {len(df):,} rows remain")
    return df.reset_index(drop=True)


def encode_ids(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Map each column's string/int values to a contiguous integer index (0-based).
    Adds columns <col>_idx alongside the originals.
    """
    for col in cols:
        uniq = sorted(df[col].astype(str).unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        df[f"{col}_idx"] = df[col].astype(str).map(mapping).astype("int32")
    return df
