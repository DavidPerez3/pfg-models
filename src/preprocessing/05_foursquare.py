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
FOURSQUARE_METADATA_CANDIDATES = [
    DATA_DIR / "foursquare_venues.csv",
    DATA_DIR / "foursquare_venues.parquet",
    DATA_DIR / "foursquare_venues.json",
    DATA_DIR / "foursquare_poi.csv",
    DATA_DIR / "foursquare_poi.parquet",
]


def parse_foursquare_ts(series: pd.Series) -> pd.Series:
    """Parse the non-standard Foursquare timestamp to unix seconds."""
    parsed = pd.to_datetime(series.str.strip(), format=TS_FORMAT, utc=True, errors="coerce")
    return (parsed.astype("int64") // 10**9).where(parsed.notna(), other=pd.NA)


def _mode_or_default(series: pd.Series, default):
    clean = series.dropna()
    if clean.empty:
        return default
    mode = clean.mode()
    if mode.empty:
        return default
    return mode.iloc[0]


def _format_profile(row: pd.Series) -> str:
    bits = [
        f"{int(row['n_checkins'])} check-ins" if pd.notna(row.get("n_checkins")) else "",
        f"{int(row['n_unique_users'])} unique users" if pd.notna(row.get("n_unique_users")) else "",
        f"peak hour {int(row['top_hour_of_day']):02d}:00" if pd.notna(row.get("top_hour_of_day")) else "",
        f"weekday {int(row['top_day_of_week'])}" if pd.notna(row.get("top_day_of_week")) else "",
        f"UTC offset {int(row['top_timezone_offset'])}" if pd.notna(row.get("top_timezone_offset")) else "",
    ]
    return " | ".join(bit for bit in bits if bit)


def load_foursquare_metadata() -> pd.DataFrame | None:
    """
    Optionally load venue-level metadata if the user later adds it to raw data.
    The current public check-in file does not include names/categories.
    """
    metadata_path = next((p for p in FOURSQUARE_METADATA_CANDIDATES if p.exists()), None)
    if metadata_path is None:
        log.warning("No Foursquare venue metadata file found. Falling back to interaction-derived item stats only.")
        return None

    log.info(f"Loading Foursquare venue metadata from {metadata_path.name} ...")
    if metadata_path.suffix == ".parquet":
        df = pd.read_parquet(metadata_path)
    elif metadata_path.suffix == ".json":
        df = pd.read_json(metadata_path, lines=True)
    else:
        df = pd.read_csv(metadata_path)

    column_aliases = {
        "venue_id": "item_id",
        "id": "item_id",
        "venue_name": "name",
        "business_name": "name",
        "category": "category",
        "categories": "categories",
        "city": "city",
        "state": "state",
        "country": "country",
        "lat": "lat",
        "latitude": "lat",
        "lon": "lon",
        "lng": "lon",
        "longitude": "lon",
    }
    renamed = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in column_aliases:
            renamed[col] = column_aliases[key]
    df = df.rename(columns=renamed)
    if "item_id" not in df.columns:
        log.warning("Foursquare venue metadata does not contain an item/venue identifier. Ignoring it.")
        return None

    keep = [c for c in ["item_id", "name", "category", "categories", "city", "state", "country", "lat", "lon"] if c in df.columns]
    df = df[keep].copy()
    df["item_id"] = df["item_id"].astype(str).str.strip()
    return df.drop_duplicates("item_id").reset_index(drop=True)


def build_items_metadata(events_df: pd.DataFrame, metadata_df: pd.DataFrame | None = None) -> pd.DataFrame:
    grouped = (
        events_df.groupby("item_id", as_index=False)
        .agg(
            n_checkins=("item_id", "size"),
            n_unique_users=("user_id", "nunique"),
            first_seen_ts=("timestamp", "min"),
            last_seen_ts=("timestamp", "max"),
            top_timezone_offset=("timezone_offset", lambda s: _mode_or_default(s, 0)),
            top_hour_of_day=("hour_of_day", lambda s: _mode_or_default(s, 0)),
            top_day_of_week=("day_of_week", lambda s: _mode_or_default(s, 0)),
            top_month=("month", lambda s: _mode_or_default(s, 0)),
        )
    )
    grouped["item_name"] = grouped["item_id"]
    grouped["title"] = grouped["item_name"]
    grouped["temporal_profile"] = grouped.apply(_format_profile, axis=1)

    if metadata_df is not None and not metadata_df.empty:
        grouped = grouped.merge(metadata_df, on="item_id", how="left")
        name_mask = grouped.get("name", pd.Series(index=grouped.index, dtype="string")).fillna("").astype(str).str.strip() != ""
        grouped.loc[name_mask, "item_name"] = grouped.loc[name_mask, "name"].astype(str)
        grouped.loc[name_mask, "title"] = grouped.loc[name_mask, "name"].astype(str)
    else:
        for col in ["name", "category", "categories", "city", "state", "country", "lat", "lon"]:
            grouped[col] = np.nan if col in {"lat", "lon"} else ""

    return grouped


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
    venue_metadata = load_foursquare_metadata()

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

    # ── Item metadata (interaction-derived stats + optional venue metadata) ───
    items = build_items_metadata(df_dedup, metadata_df=venue_metadata)

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
