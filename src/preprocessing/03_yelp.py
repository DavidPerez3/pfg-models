"""
Yelp Dataset Preprocessor
==========================
Source:
  data/raw/Yelp JSON/yelp_dataset.tar
    └── yelp_academic_dataset_review.json   (Reviews, JSON Lines)
    └── yelp_academic_dataset_business.json (Business metadata, JSON Lines)

Output:
  data/processed/yelp/interactions.parquet
  data/processed/yelp/items.parquet
"""

import io
import time
import tarfile
from pathlib import Path

import pandas as pd
import numpy as np

from utils import DATA_DIR, log, validate_schema, report_stats, save_processed, kcore_filter, encode_ids

DATASET_NAME = "yelp"
YELP_TAR = DATA_DIR / "Yelp JSON" / "yelp_dataset.tar"
CHUNK_SIZE = 200_000
K_CORE = 5

REVIEW_FILE = "yelp_academic_dataset_review.json"
BUSINESS_FILE = "yelp_academic_dataset_business.json"


def extract_member_to_df(tar: tarfile.TarFile, filename: str, chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    """Extract a JSON-Lines member from a tarball and return as DataFrame."""
    # Find the member (may be nested in a subfolder)
    member = None
    for m in tar.getmembers():
        if m.name.endswith(filename):
            member = m
            break
    if member is None:
        raise FileNotFoundError(f"{filename} not found inside tar archive")

    log.info(f"Extracting {filename} from tar …")
    f = tar.extractfile(member)
    if f is None:
        raise IOError(f"Cannot read {filename} from tar")

    # Read in chunks to avoid loading the entire file at once
    chunks = []
    buf = []
    for raw_line in f:
        buf.append(raw_line)
        if len(buf) >= chunk_size:
            chunk_bytes = b"".join(buf)
            chunks.append(pd.read_json(io.BytesIO(chunk_bytes), lines=True))
            buf = []
    if buf:
        chunk_bytes = b"".join(buf)
        chunks.append(pd.read_json(io.BytesIO(chunk_bytes), lines=True))

    df = pd.concat(chunks, ignore_index=True)
    log.info(f"  → {len(df):,} rows loaded from {filename}")
    return df


def load_reviews(tar: tarfile.TarFile) -> pd.DataFrame:
    df = extract_member_to_df(tar, REVIEW_FILE)
    df = df.rename(columns={
        "user_id": "user_id",
        "business_id": "item_id",
        "stars": "rating",
    })
    # Parse ISO datetime → unix timestamp
    df["timestamp"] = pd.to_datetime(df["date"], utc=True, errors="coerce").astype("int64") // 10**9
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("float32")
    # Keep useful feature columns
    keep = ["user_id", "item_id", "rating", "timestamp", "text", "useful", "funny", "cool"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def load_businesses(tar: tarfile.TarFile) -> pd.DataFrame:
    df = extract_member_to_df(tar, BUSINESS_FILE)
    df = df.rename(columns={"business_id": "item_id", "stars": "business_avg_stars"})
    keep = ["item_id", "name", "city", "state", "categories", "business_avg_stars"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def main():
    t0 = time.time()

    if not YELP_TAR.exists():
        log.error(f"Yelp tar not found: {YELP_TAR}")
        return

    log.info(f"Opening {YELP_TAR.name} …")
    with tarfile.open(YELP_TAR, "r:*") as tar:
        reviews = load_reviews(tar)
        businesses = load_businesses(tar)

    # ── Clean interactions ────────────────────────────────────────────────────
    reviews = reviews.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    reviews = reviews[reviews["rating"].between(1.0, 5.0)]
    reviews = reviews[reviews["timestamp"] > 0]
    log.info(f"After basic filters: {len(reviews):,} rows")

    # ── Feature engineering ──────────────────────────────────────────────────
    if "text" in reviews.columns:
        reviews["review_length"] = reviews["text"].fillna("").str.len().astype("int32")
        reviews["has_text"] = (reviews["review_length"] > 0).astype("int8")
    for col in ["useful", "funny", "cool"]:
        if col in reviews.columns:
            reviews[col] = pd.to_numeric(reviews[col], errors="coerce").fillna(0).astype("int32")
    if all(c in reviews.columns for c in ["useful", "funny", "cool"]):
        reviews["helpfulness"] = reviews["useful"] + reviews["funny"] + reviews["cool"]

    # ── k-core filtering ──────────────────────────────────────────────────────
    reviews = kcore_filter(reviews, k=K_CORE)

    reviews["dataset"] = DATASET_NAME

    # ── Encode IDs ──────────────────────────────────────────────────────────
    reviews = encode_ids(reviews, ["user_id", "item_id"])

    validate_schema(reviews, DATASET_NAME)
    report_stats(reviews, DATASET_NAME)

    # ── Item metadata ────────────────────────────────────────────────────────
    surviving_items = reviews["item_id"].unique()
    items = businesses[businesses["item_id"].isin(surviving_items)].reset_index(drop=True)

    save_processed(reviews, DATASET_NAME, extra=items)
    log.info(f"Yelp preprocessing done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
