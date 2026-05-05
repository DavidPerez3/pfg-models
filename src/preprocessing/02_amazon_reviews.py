"""
Amazon Reviews (Electronics) Preprocessor
==========================================
Source file (already 5-core):
  data/raw/Amazon Reviews/Electronics_5.json

Each file is large (~4 GB JSON Lines) — read in chunks.

Output:
  data/processed/amazon_electronics/interactions.parquet
  data/processed/amazon_electronics/items.parquet
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


def build_items_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build item-level metadata from review interactions.
    This provides items.parquet even without an external product catalog.
    """
    rows = []
    for item_id, group in df.groupby("item_id", sort=False):
        record = {
            "item_id": item_id,
            "item_name": str(item_id),
            "n_reviews": int(len(group)),
            "avg_rating": float(group["rating"].mean()),
            "rating_std": float(group["rating"].std(ddof=0)) if len(group) > 1 else 0.0,
            "timestamp_min": int(group["timestamp"].min()),
            "timestamp_max": int(group["timestamp"].max()),
        }

        if "verified" in group.columns:
            record["verified_ratio"] = float(group["verified"].astype("float32").mean())
        if "review_length" in group.columns:
            record["avg_review_length"] = float(group["review_length"].mean())
        if "vote" in group.columns:
            record["avg_vote"] = float(group["vote"].mean())
        if "style_str" in group.columns:
            styles = group["style_str"].dropna().astype(str)
            record["top_style"] = styles.mode().iloc[0] if not styles.empty else ""
        if "summary" in group.columns:
            summaries = group["summary"].dropna().astype(str).str.strip()
            summaries = summaries[summaries != ""]
            record["sample_summary"] = summaries.iloc[0] if not summaries.empty else ""

        rows.append(record)

    items = pd.DataFrame(rows)
    for col in ["verified_ratio", "avg_review_length", "avg_vote", "top_style", "sample_summary"]:
        if col not in items.columns:
            items[col] = np.nan if col not in {"top_style", "sample_summary"} else ""
    return items


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

    items = build_items_metadata(df)
    if "item_id_idx" in df.columns:
        item_idx_map = (
            df[["item_id", "item_id_idx"]]
            .drop_duplicates("item_id")
            .set_index("item_id")["item_id_idx"]
        )
        items["item_id_idx"] = items["item_id"].map(item_idx_map).astype("int32")

    validate_schema(df, name)
    report_stats(df, name)
    save_processed(df, name, extra=items)
    log.info(f"{name} done in {time.time() - t0:.1f}s")


def main():
    for name, path in DATASETS.items():
        if not path.exists():
            log.error(f"File not found: {path}")
            continue
        process_amazon(name, path)


if __name__ == "__main__":
    main()
