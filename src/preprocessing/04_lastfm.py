"""
Last.fm 1K Preprocessor
=======================
Supported source format (typically no header):
  data/raw/userid-timestamp-artid-artname-traid-traname.tsv

Canonical columns:
  user_id, timestamp, artist_id, artist_name, track_id, track_name

Output:
  data/processed/lastfm/interactions.parquet
  data/processed/lastfm/items.parquet
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils import DATA_DIR, log, validate_schema, report_stats, save_processed, encode_ids

DATASET_NAME = "lastfm"
LASTFM_CANDIDATE_FILES = [
    DATA_DIR / "userid-timestamp-artid-artname-traid-traname.tsv",
    DATA_DIR / "lastfm-dataset-1K" / "userid-timestamp-artid-artname-traid-traname.tsv",
    DATA_DIR / "Last.fm_data.tsv",
]

# Default to artist-level items for robustness against track-level sparsity.
ITEM_LEVEL = "artist"  # {"artist", "track"}
MIN_USER_INTERACTIONS = 3
MIN_ITEM_INTERACTIONS = 3
CHUNK_SIZE = 2_000_000

EXPECTED_COLS = ["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"]


def resolve_input_file() -> Path:
    for p in LASTFM_CANDIDATE_FILES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Last.fm 1K file not found. Expected one of:\n"
        + "\n".join(f"  - {p}" for p in LASTFM_CANDIDATE_FILES)
    )


def detect_format(path: Path) -> tuple[str, bool]:
    """
    Detect separator and whether the first line is a header.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    first_line = ""
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                first_line = f.readline().strip("\n\r")
            break
        except Exception as e:  # pragma: no cover
            last_err = e
            continue
    if not first_line:
        raise RuntimeError(f"Could not read first line from {path}: {last_err}")

    sep = "\t" if first_line.count("\t") >= first_line.count(",") else ","
    tokens = [t.strip().lower() for t in first_line.split(sep)]
    header_tokens = {"user_id", "userid", "timestamp", "artid", "artist_id", "traid", "track_id"}
    has_header = len(header_tokens.intersection(tokens)) >= 2
    return sep, has_header


def normalize_column_names(cols: list[str]) -> list[str]:
    """
    Map possible Last.fm 1K header variants to canonical names.
    """
    mapped = []
    for c in cols:
        key = c.strip().lower().replace("-", "_")
        if key in {"userid", "user", "user_id"}:
            mapped.append("user_id")
        elif key in {"timestamp", "time"}:
            mapped.append("timestamp")
        elif key in {"artid", "artistid", "artist_id"}:
            mapped.append("artist_id")
        elif key in {"artname", "artistname", "artist_name"}:
            mapped.append("artist_name")
        elif key in {"traid", "trackid", "track_id"}:
            mapped.append("track_id")
        elif key in {"traname", "trackname", "track_name"}:
            mapped.append("track_name")
        else:
            mapped.append(key)
    return mapped


def parse_timestamp(ts: pd.Series) -> pd.Series:
    """
    Parse timestamp column into unix seconds.
    Supports ISO timestamps and numeric unix values.
    """
    ts_str = ts.astype("string").str.strip()
    dt = pd.to_datetime(ts_str, utc=True, errors="coerce")
    out = (dt.astype("int64") // 10**9).where(dt.notna(), other=pd.NA)

    numeric_ts = pd.to_numeric(ts_str, errors="coerce")
    # If parsing as datetime fails but numeric exists, use numeric as unix timestamp.
    fill_mask = out.isna() & numeric_ts.notna()
    out = out.mask(fill_mask, numeric_ts)
    return pd.to_numeric(out, errors="coerce").astype("Int64")


def build_item_columns(df: pd.DataFrame, item_level: str = ITEM_LEVEL) -> pd.DataFrame:
    """
    Build item_id/item_name with explicit fallback strategy:
    - Prefer stable IDs (artist_id/track_id)
    - If missing, fallback to names
    - If still missing, row is dropped later
    """
    for col in ["artist_id", "artist_name", "track_id", "track_name", "user_id"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace("", pd.NA)

    if item_level == "track":
        df["item_id"] = df["track_id"].fillna("track_name::" + df["track_name"].fillna(""))
        missing_track = df["item_id"].isna() | (df["item_id"].astype("string").str.strip() == "track_name::")
        if missing_track.any():
            fallback = "track_comp::" + df["artist_name"].fillna("") + "|||" + df["track_name"].fillna("")
            df.loc[missing_track, "item_id"] = fallback.loc[missing_track]
        df["item_name"] = df["track_name"].fillna(df["artist_name"])
    else:
        df["item_id"] = df["artist_id"].fillna("artist_name::" + df["artist_name"].fillna(""))
        df["item_name"] = df["artist_name"]

    df["item_id"] = df["item_id"].astype("string").str.strip().replace("", pd.NA)
    df["item_name"] = df["item_name"].astype("string").str.strip().replace("", pd.NA)
    return df


def aggregate_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate implicit events into user-item interactions for one chunk.
    """
    out = (
        df.groupby(["user_id", "item_id"], dropna=False)
        .agg(
            num_events=("item_id", "size"),
            timestamp_min=("timestamp", "min"),
            timestamp_max=("timestamp", "max"),
            item_name=("item_name", "first"),
        )
        .reset_index()
    )
    return out


def kcore_filter_parametrized(
    df: pd.DataFrame,
    min_user_interactions: int = MIN_USER_INTERACTIONS,
    min_item_interactions: int = MIN_ITEM_INTERACTIONS,
) -> pd.DataFrame:
    """
    Iteratively remove users/items below per-side thresholds.
    """
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_user_interactions].index)]
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= min_item_interactions].index)]
    log.info(
        "After core filtering (min_user=%d, min_item=%d): %s rows",
        min_user_interactions,
        min_item_interactions,
        f"{len(df):,}",
    )
    return df.reset_index(drop=True)


def report_distribution_stats(df: pd.DataFrame) -> None:
    """
    Print basic distribution stats requested for monitoring.
    """
    user_counts = df.groupby("user_id").size()
    item_counts = df.groupby("item_id").size()
    event_counts = df["num_events"] if "num_events" in df.columns else pd.Series(dtype="int64")

    def q(series: pd.Series, p: float) -> float:
        return float(series.quantile(p)) if len(series) else float("nan")

    log.info("Distribution stats:")
    log.info(
        "  interactions/user  p50=%.1f p90=%.1f p99=%.1f max=%s",
        q(user_counts, 0.50),
        q(user_counts, 0.90),
        q(user_counts, 0.99),
        int(user_counts.max()) if len(user_counts) else 0,
    )
    log.info(
        "  interactions/item  p50=%.1f p90=%.1f p99=%.1f max=%s",
        q(item_counts, 0.50),
        q(item_counts, 0.90),
        q(item_counts, 0.99),
        int(item_counts.max()) if len(item_counts) else 0,
    )
    if len(event_counts):
        log.info(
            "  events per pair    p50=%.1f p90=%.1f p99=%.1f max=%s",
            q(event_counts, 0.50),
            q(event_counts, 0.90),
            q(event_counts, 0.99),
            int(event_counts.max()),
        )


def main():
    t0 = time.time()
    source_file = resolve_input_file()
    sep, has_header = detect_format(source_file)

    log.info("Loading Last.fm 1K source: %s", source_file)
    log.info("Detected format: sep=%r, header=%s", sep, has_header)
    log.info("Item granularity: %s", ITEM_LEVEL)

    read_kwargs = {
        "sep": sep,
        "dtype": "string",
        "na_values": ["", "NA", "NaN", "NULL", "null"],
        "encoding": "utf-8",
        "chunksize": CHUNK_SIZE,
        # Some Last.fm 1K dumps contain malformed lines with extra delimiters.
        # Skip those lines instead of crashing the whole preprocessing.
        "on_bad_lines": "skip",
    }
    if has_header:
        read_kwargs["header"] = 0
    else:
        read_kwargs["header"] = None
        read_kwargs["names"] = EXPECTED_COLS

    raw_rows = 0
    valid_rows_after_clean = 0
    chunk_aggs = []

    try:
        chunk_iter = pd.read_csv(source_file, **read_kwargs)
        for idx, chunk in enumerate(chunk_iter, start=1):
            if has_header and idx == 1:
                chunk.columns = normalize_column_names(list(chunk.columns))

            raw_rows += len(chunk)

            # Keep only known fields if extra columns are present.
            keep_cols = [c for c in EXPECTED_COLS if c in chunk.columns]
            chunk = chunk[keep_cols].copy()
            for c in EXPECTED_COLS:
                if c not in chunk.columns:
                    chunk[c] = pd.NA

            chunk = build_item_columns(chunk, item_level=ITEM_LEVEL)
            chunk["timestamp"] = parse_timestamp(chunk["timestamp"])

            chunk = chunk.dropna(subset=["user_id", "item_id", "timestamp"])
            chunk = chunk[chunk["timestamp"] > 0]
            valid_rows_after_clean += len(chunk)

            if len(chunk):
                chunk_aggs.append(aggregate_chunk(chunk))

            if idx % 5 == 0:
                log.info(
                    "Processed chunks=%d | raw_rows=%s | cleaned_rows=%s",
                    idx,
                    f"{raw_rows:,}",
                    f"{valid_rows_after_clean:,}",
                )

    except pd.errors.ParserError:
        # Retry with the python engine (more tolerant) while keeping bad-line skipping.
        log.warning("ParserError with default CSV engine. Retrying with engine='python' and on_bad_lines='skip'.")
        read_kwargs["engine"] = "python"
        chunk_iter = pd.read_csv(source_file, **read_kwargs)
        for idx, chunk in enumerate(chunk_iter, start=1):
            if has_header and idx == 1:
                chunk.columns = normalize_column_names(list(chunk.columns))

            raw_rows += len(chunk)

            keep_cols = [c for c in EXPECTED_COLS if c in chunk.columns]
            chunk = chunk[keep_cols].copy()
            for c in EXPECTED_COLS:
                if c not in chunk.columns:
                    chunk[c] = pd.NA

            chunk = build_item_columns(chunk, item_level=ITEM_LEVEL)
            chunk["timestamp"] = parse_timestamp(chunk["timestamp"])

            chunk = chunk.dropna(subset=["user_id", "item_id", "timestamp"])
            chunk = chunk[chunk["timestamp"] > 0]
            valid_rows_after_clean += len(chunk)

            if len(chunk):
                chunk_aggs.append(aggregate_chunk(chunk))

            if idx % 5 == 0:
                log.info(
                    "Processed chunks=%d | raw_rows=%s | cleaned_rows=%s",
                    idx,
                    f"{raw_rows:,}",
                    f"{valid_rows_after_clean:,}",
                )

    except UnicodeDecodeError:
        # Fallback for non-utf8 files.
        read_kwargs["encoding"] = "latin-1"
        chunk_iter = pd.read_csv(source_file, **read_kwargs)
        for idx, chunk in enumerate(chunk_iter, start=1):
            if has_header and idx == 1:
                chunk.columns = normalize_column_names(list(chunk.columns))
            raw_rows += len(chunk)
            keep_cols = [c for c in EXPECTED_COLS if c in chunk.columns]
            chunk = chunk[keep_cols].copy()
            for c in EXPECTED_COLS:
                if c not in chunk.columns:
                    chunk[c] = pd.NA
            chunk = build_item_columns(chunk, item_level=ITEM_LEVEL)
            chunk["timestamp"] = parse_timestamp(chunk["timestamp"])
            chunk = chunk.dropna(subset=["user_id", "item_id", "timestamp"])
            chunk = chunk[chunk["timestamp"] > 0]
            valid_rows_after_clean += len(chunk)
            if len(chunk):
                chunk_aggs.append(aggregate_chunk(chunk))

    if not chunk_aggs:
        raise RuntimeError("No valid interactions could be built from Last.fm source.")

    log.info("Merging chunk-level aggregates ...")
    interactions = pd.concat(chunk_aggs, ignore_index=True)
    interactions = (
        interactions.groupby(["user_id", "item_id"], as_index=False)
        .agg(
            num_events=("num_events", "sum"),
            timestamp_min=("timestamp_min", "min"),
            timestamp_max=("timestamp_max", "max"),
            item_name=("item_name", "first"),
        )
    )

    # Align with pipeline schema:
    # - timestamp: use latest event (timestamp_max)
    # - rating: implicit strength derived from play count (bounded for schema validation).
    interactions["timestamp"] = interactions["timestamp_max"].astype("int64")
    interactions["rating"] = np.log1p(interactions["num_events"].astype("float32")).clip(0, 5)
    interactions["dataset"] = DATASET_NAME

    log.info("Rows after aggregation (user-item): %s", f"{len(interactions):,}")

    interactions = kcore_filter_parametrized(
        interactions,
        min_user_interactions=MIN_USER_INTERACTIONS,
        min_item_interactions=MIN_ITEM_INTERACTIONS,
    )
    interactions = encode_ids(interactions, ["user_id", "item_id"])

    validate_schema(interactions, DATASET_NAME)
    report_stats(interactions, DATASET_NAME)

    n_users = interactions["user_id"].nunique()
    n_items = interactions["item_id"].nunique()
    n_interactions = len(interactions)
    sparsity = 1 - n_interactions / (n_users * n_items) if n_users and n_items else float("nan")
    log.info("Summary control stats:")
    log.info("  raw rows                     : %s", f"{raw_rows:,}")
    log.info("  rows after cleaning          : %s", f"{valid_rows_after_clean:,}")
    log.info("  users                        : %s", f"{n_users:,}")
    log.info("  items                        : %s", f"{n_items:,}")
    log.info("  final interactions           : %s", f"{n_interactions:,}")
    log.info("  sparsity                     : %.6f", sparsity)
    report_distribution_stats(interactions)

    items = interactions[["item_id", "item_name"]].drop_duplicates("item_id").reset_index(drop=True)

    interaction_cols = [
        "user_id",
        "item_id",
        "rating",
        "timestamp",
        "dataset",
        "num_events",
        "timestamp_min",
        "timestamp_max",
        "user_id_idx",
        "item_id_idx",
    ]
    interaction_cols = [c for c in interaction_cols if c in interactions.columns]
    save_processed(interactions[interaction_cols], DATASET_NAME, extra=items)
    log.info("Last.fm preprocessing done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
