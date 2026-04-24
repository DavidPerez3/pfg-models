"""
Foursquare Preprocessor
========================
Source:
  data/raw/foursquare.txt
    Tab-separated, NO header row.
    Columns: user_id | venue_id | timestamp_str | timezone_offset
    Example: 50756  4f5e3a72e4b053fd6a4313f6  Tue Apr 03 18:00:06 +0000 2012  240

Feedback type: IMPLICIT — each row is a check-in event.
  rating = 1.0 (binary implicit)
  check_in_count kept as auxiliary feature per (user, venue)

Output:
  data/processed/foursquare/interactions.parquet
  data/processed/foursquare/items.parquet
"""

import time
import pandas as pd
import numpy as np

from utils import DATA_DIR, log, validate_schema, report_stats, save_processed, kcore_filter, encode_ids

DATASET_NAME = "foursquare"
FOURSQUARE_TXT = DATA_DIR / "foursquare.txt"
K_CORE = 5
CHUNK_SIZE = 500_000

# Timestamp format in the file: "Tue Apr 03 18:00:06 +0000 2012"
TS_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def parse_foursquare_ts(series: pd.Series) -> pd.Series:
    """Parse the non-standard Foursquare timestamp to unix seconds."""
    parsed = pd.to_datetime(series.str.strip(), format=TS_FORMAT, utc=True, errors="coerce")
    return (parsed.astype("int64") // 10**9).where(parsed.notna(), other=pd.NA)


def main():
    t0 = time.time()

    log.info(f"Reading {FOURSQUARE_TXT.name} in chunks of {CHUNK_SIZE:,} …")

    col_names = ["user_id", "venue_id", "timestamp_str", "timezone_offset"]
    chunks = []
    reader = pd.read_csv(
        FOURSQUARE_TXT,
        sep="\t",
        header=None,
        names=col_names,
        dtype={"user_id": str, "venue_id": str, "timezone_offset": str},
        chunksize=CHUNK_SIZE,
        encoding="utf-8",
        on_bad_lines="skip",
    )
    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        if (i + 1) % 5 == 0:
            log.info(f"  Read {(i + 1) * CHUNK_SIZE:,} rows …")

    df = pd.concat(chunks, ignore_index=True)
    log.info(f"Loaded {len(df):,} rows total")

    # ── Rename to unified schema ────────────────────────────────────────────
    df = df.rename(columns={"venue_id": "item_id"})

    # ── Drop completely null rows ─────────────────────────────────────────────
    df = df.dropna(subset=["user_id", "item_id", "timestamp_str"])
    log.info(f"After dropping nulls: {len(df):,} rows")

    # ── Parse timestamps ──────────────────────────────────────────────────────
    df["timestamp"] = parse_foursquare_ts(df["timestamp_str"])
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].astype("int64")
    log.info(f"After timestamp parse: {len(df):,} rows")

    # ── Timezone offset (keep as integer minutes) ────────────────────────────
    df["timezone_offset"] = pd.to_numeric(df["timezone_offset"], errors="coerce").fillna(0).astype("int16")

    # ── Temporal features ─────────────────────────────────────────────────────
    dt = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["hour_of_day"] = dt.dt.hour.astype("int8")
    df["day_of_week"] = dt.dt.dayofweek.astype("int8")
    df["month"] = dt.dt.month.astype("int8")

    # ── Compute check-in count per (user, venue) ──────────────────────────────
    checkin_counts = (
        df.groupby(["user_id", "item_id"])
        .size()
        .reset_index(name="check_in_count")
    )

    # ── Deduplicate: one row per (user, item), keep earliest timestamp ────────
    df_dedup = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["user_id", "item_id"], keep="first")
        .reset_index(drop=True)
    )
    log.info(f"After deduplication: {len(df_dedup):,} rows")

    # Merge check_in_count back
    df_dedup = df_dedup.merge(checkin_counts, on=["user_id", "item_id"], how="left")

    # ── Implicit rating ────────────────────────────────────────────────────────
    df_dedup["rating"] = 1.0
    df_dedup["dataset"] = DATASET_NAME

    # ── k-core filtering ──────────────────────────────────────────────────────
    df_dedup = kcore_filter(df_dedup, k=K_CORE)

    # ── Encode IDs ──────────────────────────────────────────────────────────
    df_dedup = encode_ids(df_dedup, ["user_id", "item_id"])

    validate_schema(df_dedup, DATASET_NAME)
    report_stats(df_dedup, DATASET_NAME)

    # ── Item metadata (just the unique venue IDs with temporal stats) ─────────
    items = df_dedup[["item_id"]].drop_duplicates().reset_index(drop=True)

    interaction_cols = [
        "user_id", "item_id", "rating", "timestamp", "dataset",
        "check_in_count", "timezone_offset",
        "hour_of_day", "day_of_week", "month",
        "user_id_idx", "item_id_idx",
    ]
    interaction_cols = [c for c in interaction_cols if c in df_dedup.columns]
    save_processed(df_dedup[interaction_cols], DATASET_NAME, extra=items)
    log.info(f"Foursquare preprocessing done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
