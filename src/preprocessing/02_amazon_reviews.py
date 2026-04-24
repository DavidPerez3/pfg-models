"""
Amazon Reviews (Electronics) Preprocessor
==========================================
Source file (already 5-core):
  data/raw/Amazon Reviews/Electronics_5.json

Each file is large (~4 GB JSON Lines) — read in chunks.

Output:
  data/processed/amazon_electronics/interactions.parquet
"""

import time
import ast
import pandas as pd
import numpy as np
from pathlib import Path

from utils import DATA_DIR, log, validate_schema, report_stats, save_processed, encode_ids

AMAZON_DIR = DATA_DIR / "Amazon Reviews"
CHUNK_SIZE = 200_000

DATASETS = {
    "amazon_electronics": AMAZON_DIR / "Electronics_5.json",
}


def read_jsonlines_chunked(path: Path) -> pd.DataFrame:
    """Read a large JSON-Lines file in chunks and concatenate."""
    chunks = []
    log.info(f"Reading {path.name} in chunks of {CHUNK_SIZE:,} …")
    for chunk in pd.read_json(path, lines=True, chunksize=CHUNK_SIZE):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    log.info(f"  → {len(df):,} rows loaded")
    return df


def parse_style(style_val) -> str:
    """Extract a flat string from the 'style' dict column (e.g. {'Format:': 'DVD'})."""
    if pd.isna(style_val) or style_val is None:
        return ""
    if isinstance(style_val, dict):
        return "; ".join(f"{k.strip(':')}: {v}" for k, v in style_val.items())
    try:
        d = ast.literal_eval(str(style_val))
        if isinstance(d, dict):
            return "; ".join(f"{k.strip(':')}: {v}" for k, v in d.items())
    except Exception:
        pass
    return str(style_val)


def process_amazon(name: str, path: Path):
    t0 = time.time()
    log.info(f"\n{'='*60}")
    log.info(f"Processing {name} from {path.name}")

    df = read_jsonlines_chunked(path)

    # ── Rename to unified schema ────────────────────────────────────────────
    df = df.rename(columns={
        "reviewerID": "user_id",
        "asin": "item_id",
        "overall": "rating",
        "unixReviewTime": "timestamp",
    })

    # ── Select & clean columns ──────────────────────────────────────────────
    keep_cols = ["user_id", "item_id", "rating", "timestamp"]
    extra_cols = [c for c in ["reviewText", "summary", "vote", "style", "verified"] if c in df.columns]
    df = df[keep_cols + extra_cols].copy()

    # Rating: must be in [1, 5]
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("float32")
    df = df[df["rating"].between(1.0, 5.0)]

    # Timestamp: must be positive integer
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    df["timestamp"] = df["timestamp"].astype("int64")

    # ── Feature engineering ──────────────────────────────────────────────────
    if "reviewText" in df.columns:
        df["review_length"] = df["reviewText"].fillna("").str.len().astype("int32")
        df["has_text"] = (df["review_length"] > 0).astype("int8")
    if "vote" in df.columns:
        # vote can be a string like "1,234" → convert
        df["vote"] = df["vote"].astype(str).str.replace(",", "").str.strip()
        df["vote"] = pd.to_numeric(df["vote"], errors="coerce").fillna(0).astype("int32")
    if "style" in df.columns:
        df["style_str"] = df["style"].apply(parse_style)
        df = df.drop(columns=["style"])
    if "verified" in df.columns:
        df["verified"] = df["verified"].astype("bool")

    # ── No k-core needed (files are already 5-core) ──────────────────────────
    df["dataset"] = name

    # ── Encode IDs ──────────────────────────────────────────────────────────
    df = encode_ids(df, ["user_id", "item_id"])

    validate_schema(df, name)
    report_stats(df, name)
    save_processed(df, name)
    log.info(f"{name} done in {time.time() - t0:.1f}s")


def main():
    for name, path in DATASETS.items():
        if not path.exists():
            log.error(f"File not found: {path}")
            continue
        process_amazon(name, path)


if __name__ == "__main__":
    main()
