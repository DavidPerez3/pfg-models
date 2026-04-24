"""
MovieLens 20M Preprocessor
===========================
Source files:
  data/raw/ml-20m/ratings.csv     → userId, movieId, rating, timestamp
  data/raw/ml-20m/movies.csv      → movieId, title, genres
  data/raw/ml-20m/genome-scores.csv  → movieId, tagId, relevance
  data/raw/ml-20m/genome-tags.csv    → tagId, tag

Output:
  data/processed/movielens/interactions.parquet
  data/processed/movielens/items.parquet
"""

import time
import pandas as pd
import numpy as np

from utils import DATA_DIR, log, validate_schema, report_stats, save_processed, kcore_filter, encode_ids

DATASET_NAME = "movielens"
ML_DIR = DATA_DIR / "ml-20m"
K_CORE = 5
TOP_TAGS = 20  # Number of genome tags to pivot as item features


def load_ratings() -> pd.DataFrame:
    log.info("Loading ratings.csv …")
    df = pd.read_csv(ML_DIR / "ratings.csv", dtype={"userId": str, "movieId": str})
    df = df.rename(columns={"userId": "user_id", "movieId": "item_id", "rating": "rating", "timestamp": "timestamp"})
    df["rating"] = df["rating"].astype("float32")
    df["timestamp"] = df["timestamp"].astype("int64")
    df["dataset"] = DATASET_NAME
    return df


def load_movies() -> pd.DataFrame:
    log.info("Loading movies.csv …")
    movies = pd.read_csv(ML_DIR / "movies.csv", dtype={"movieId": str})
    movies = movies.rename(columns={"movieId": "item_id"})
    # Split genres pipe-separated string into a list kept as string (compatible with Parquet)
    movies["genre_list"] = movies["genres"].str.split("|")
    movies["genres_str"] = movies["genres"]  # keep original
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype("float32")
    return movies[["item_id", "title", "genres_str", "genre_list", "year"]]


def load_genome(top_n: int = TOP_TAGS) -> pd.DataFrame:
    """
    Return a wide DataFrame: item_id | tag_<name> columns.
    We take the top_n most informative tags (highest mean relevance across movies).
    """
    log.info("Loading genome-scores.csv …")
    scores = pd.read_csv(ML_DIR / "genome-scores.csv", dtype={"movieId": str})
    scores = scores.rename(columns={"movieId": "item_id"})

    log.info("Loading genome-tags.csv …")
    tags = pd.read_csv(ML_DIR / "genome-tags.csv")
    tags["tag_col"] = "tag_" + tags["tag"].str.lower().str.replace(r"\W+", "_", regex=True)

    # Find top_n most discriminative tags by variance of relevance
    avg = scores.groupby("tagId")["relevance"].var().nlargest(top_n)
    top_tag_ids = avg.index.tolist()

    scores_top = scores[scores["tagId"].isin(top_tag_ids)]
    # Map tagId → clean column name
    id2name = dict(zip(tags["tagId"], tags["tag_col"]))
    scores_top = scores_top.copy()
    scores_top["tag_name"] = scores_top["tagId"].map(id2name)

    # Pivot to wide format
    genome_wide = scores_top.pivot_table(index="item_id", columns="tag_name", values="relevance")
    genome_wide = genome_wide.reset_index()
    genome_wide.columns.name = None
    return genome_wide


def main():
    t0 = time.time()

    # ── Interactions ────────────────────────────────────────────────────────
    interactions = load_ratings()
    log.info(f"Raw interactions: {len(interactions):,} rows")

    # Drop nulls in mandatory columns
    interactions = interactions.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    # Validate rating range
    interactions = interactions[interactions["rating"].between(0.5, 5.0)]
    log.info(f"After basic filters: {len(interactions):,} rows")

    # k-core filtering
    interactions = kcore_filter(interactions, k=K_CORE)

    # Encode IDs
    interactions = encode_ids(interactions, ["user_id", "item_id"])

    validate_schema(interactions, DATASET_NAME)
    report_stats(interactions, DATASET_NAME)

    # ── Item metadata ────────────────────────────────────────────────────────
    movies = load_movies()
    try:
        genome = load_genome()
        items = movies.merge(genome, on="item_id", how="left")
    except Exception as e:
        log.warning(f"Genome merge failed ({e}), skipping genome features.")
        items = movies

    # Keep only items that survived k-core
    surviving_items = interactions["item_id"].unique()
    items = items[items["item_id"].isin(surviving_items)].reset_index(drop=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    save_processed(interactions, DATASET_NAME, extra=items)
    log.info(f"MovieLens preprocessing done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
