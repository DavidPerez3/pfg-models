"""
06_embed_and_index.py - Text Embedding + Elasticsearch Indexer
===============================================================
Reads items.parquet from each preprocessed dataset, generates text
embeddings using sentence-transformers, and indexes into Elasticsearch.

Index structure per dataset:
  pfg_<dataset_name>_items         -> item catalog + text embedding
  pfg_<dataset_name>_interactions  -> user-item interactions (no vector)

Behavior on rerun:
  - Uses deterministic Elasticsearch _id values (idempotent, no duplicates).
  - Stores local checkpoints per dataset/section to resume after Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROCESSED_DIR, log

# Config
ES_HOST = "http://localhost:9200"
EMBED_MODEL = "all-MiniLM-L6-v2"  # 384 dims
EMBED_DIM = 384
BATCH_SIZE = 256
BULK_CHUNK = 500
INDEX_PREFIX = "pfg"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / ".index_checkpoints"
PROGRESS_MIN_INTERVAL = 3.0

# Silence verbose Elasticsearch request logs that make tqdm output messy.
logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)


# Dataset-specific text builders
def text_movielens(row: pd.Series) -> str:
    parts = [str(row.get("title", ""))]
    genres = row.get("genres_str", "") or ""
    if genres and genres != "(no genres listed)":
        parts.append(genres.replace("|", " "))
    return " | ".join(p for p in parts if p)


def text_amazon(row: pd.Series) -> str:
    return str(row.get("item_id", ""))


def text_yelp(row: pd.Series) -> str:
    parts = [
        str(row.get("name", "")),
        str(row.get("categories", "") or ""),
        str(row.get("city", "") or ""),
        str(row.get("state", "") or ""),
    ]
    return " | ".join(p for p in parts if p and p != "nan")


def text_lastfm(row: pd.Series) -> str:
    parts = [
        str(row.get("artist_name", "")),
        str(row.get("track_name", "")),
        str(row.get("album_name", "") or ""),
    ]
    return " - ".join(p for p in parts if p and p != "nan")


def text_foursquare(row: pd.Series) -> str:
    return str(row.get("item_id", ""))


TEXT_BUILDERS = {
    "movielens": text_movielens,
    "amazon_movies": text_amazon,
    "amazon_electronics": text_amazon,
    "yelp": text_yelp,
    "lastfm": text_lastfm,
    "foursquare": text_foursquare,
}


# Index mappings
def items_mapping() -> dict:
    return {
        "mappings": {
            "properties": {
                "item_id": {"type": "keyword"},
                "dataset": {"type": "keyword"},
                "text_repr": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBED_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
                "title": {"type": "text"},
                "genres": {"type": "keyword"},
                "artist": {"type": "keyword"},
                "track": {"type": "keyword"},
                "categories": {"type": "keyword"},
                "city": {"type": "keyword"},
                "avg_rating": {"type": "float"},
                "year": {"type": "float"},
            }
        },
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    }


def interactions_mapping() -> dict:
    return {
        "mappings": {
            "properties": {
                "user_id": {"type": "keyword"},
                "item_id": {"type": "keyword"},
                "rating": {"type": "float"},
                "timestamp": {"type": "date", "format": "epoch_second"},
                "dataset": {"type": "keyword"},
                "play_count": {"type": "integer"},
                "check_in_count": {"type": "integer"},
                "review_length": {"type": "integer"},
                "vote": {"type": "integer"},
                "hour_of_day": {"type": "byte"},
                "day_of_week": {"type": "byte"},
                "month": {"type": "byte"},
            }
        },
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    }


# Checkpoint helpers
def checkpoint_path(dataset_name: str, section: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{dataset_name}_{section}.ckpt"


def load_checkpoint(dataset_name: str, section: str) -> int:
    path = checkpoint_path(dataset_name, section)
    if not path.exists():
        return 0
    try:
        return max(0, int(path.read_text(encoding="utf-8").strip() or "0"))
    except ValueError:
        return 0


def save_checkpoint(dataset_name: str, section: str, value: int) -> None:
    checkpoint_path(dataset_name, section).write_text(
        str(max(0, int(value))), encoding="utf-8"
    )


def clear_checkpoint(dataset_name: str, section: str) -> None:
    path = checkpoint_path(dataset_name, section)
    if path.exists():
        path.unlink()


# Elasticsearch helpers
def get_es_client(es_host: str) -> Elasticsearch:
    es = Elasticsearch(es_host, request_timeout=60)
    info = es.info()
    log.info(f"Connected to Elasticsearch {info['version']['number']} at {es_host}")
    return es


def ensure_index(es: Elasticsearch, index: str, mapping: dict, rebuild: bool = False) -> bool:
    exists = es.indices.exists(index=index)
    if exists:
        if rebuild:
            log.info(f"Deleting existing index: {index}")
            es.indices.delete(index=index)
            exists = False
        else:
            log.info(f"Index already exists (use --rebuild to recreate): {index}")
            return True
    if not exists:
        log.info(f"Creating index: {index}")
        es.indices.create(index=index, body=mapping)
    return False


def bulk_index(es: Elasticsearch, actions: list[dict]) -> int:
    ok, errors = helpers.bulk(es, actions, chunk_size=BULK_CHUNK, raise_on_error=False)
    if errors:
        log.warning(f"  {len(errors)} bulk errors (first: {errors[0]})")
    return ok


# Per-dataset processing
def index_items(
    es: Elasticsearch,
    model: SentenceTransformer,
    dataset_name: str,
    items_parquet: Path,
    items_index: str,
    rebuild: bool,
) -> None:
    section = "items"
    index_preexisted = ensure_index(es, items_index, items_mapping(), rebuild=rebuild)
    if rebuild or not index_preexisted:
        clear_checkpoint(dataset_name, section)

    items_df = pd.read_parquet(items_parquet)
    log.info(f"  Items: {len(items_df):,} rows")

    start_row = load_checkpoint(dataset_name, section)
    if start_row > len(items_df):
        start_row = 0
        clear_checkpoint(dataset_name, section)
    if start_row:
        log.info(f"  Resuming items from row {start_row:,}")

    text_fn = TEXT_BUILDERS.get(dataset_name, lambda r: str(r.get("item_id", "")))

    total_indexed = 0
    for start in tqdm(
        range(start_row, len(items_df), BATCH_SIZE),
        desc=f"{dataset_name} items",
        unit="batch",
        mininterval=PROGRESS_MIN_INTERVAL,
        maxinterval=10.0,
        dynamic_ncols=True,
    ):
        batch = items_df.iloc[start : start + BATCH_SIZE]
        texts = [text_fn(row) for _, row in batch.iterrows()]
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)

        actions = []
        for (_, row), text, emb in zip(batch.iterrows(), texts, embeddings):
            item_id = str(row.get("item_id", ""))
            doc = {
                "item_id": item_id,
                "dataset": dataset_name,
                "text_repr": text,
                "embedding": emb.tolist(),
            }
            for src, dst in [
                ("title", "title"),
                ("genres_str", "genres"),
                ("artist_name", "artist"),
                ("track_name", "track"),
                ("categories", "categories"),
                ("city", "city"),
                ("business_avg_stars", "avg_rating"),
                ("year", "year"),
            ]:
                val = row.get(src)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    doc[dst] = val

            actions.append(
                {
                    "_op_type": "index",
                    "_index": items_index,
                    "_id": f"{dataset_name}::{item_id}",
                    "_source": doc,
                }
            )

        indexed = bulk_index(es, actions)
        total_indexed += indexed
        save_checkpoint(dataset_name, section, start + len(batch))

    log.info(f"  Total items indexed this run: {total_indexed:,}")
    save_checkpoint(dataset_name, section, len(items_df))


def index_interactions(
    es: Elasticsearch,
    dataset_name: str,
    interactions_parquet: Path,
    inter_index: str,
    rebuild: bool,
) -> None:
    section = "interactions"
    index_preexisted = ensure_index(es, inter_index, interactions_mapping(), rebuild=rebuild)
    if rebuild or not index_preexisted:
        clear_checkpoint(dataset_name, section)

    inter_df = pd.read_parquet(interactions_parquet)
    log.info(f"  Interactions: {len(inter_df):,} rows")

    start_row = load_checkpoint(dataset_name, section)
    if start_row > len(inter_df):
        start_row = 0
        clear_checkpoint(dataset_name, section)
    if start_row:
        log.info(f"  Resuming interactions from row {start_row:,}")

    cols_to_drop = [c for c in inter_df.columns if c.endswith("_idx")]
    inter_df = inter_df.drop(columns=cols_to_drop, errors="ignore")

    total_indexed = 0
    for start in tqdm(
        range(start_row, len(inter_df), BULK_CHUNK),
        desc=f"{dataset_name} interactions",
        unit="chunk",
        mininterval=PROGRESS_MIN_INTERVAL,
        maxinterval=10.0,
        dynamic_ncols=True,
    ):
        chunk = inter_df.iloc[start : start + BULK_CHUNK]
        actions = []

        for row_pos, (_, row) in enumerate(chunk.iterrows(), start=start):
            doc = {
                k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in row.items()
                if v is not None
            }
            doc["user_id"] = str(doc.get("user_id", ""))
            doc["item_id"] = str(doc.get("item_id", ""))
            doc["rating"] = float(doc.get("rating", 0))
            doc["dataset"] = dataset_name

            actions.append(
                {
                    "_op_type": "index",
                    "_index": inter_index,
                    "_id": f"{dataset_name}::{row_pos}",
                    "_source": doc,
                }
            )

        total_indexed += bulk_index(es, actions)
        save_checkpoint(dataset_name, section, start + len(chunk))

    log.info(f"  Total interactions indexed this run: {total_indexed:,}")
    save_checkpoint(dataset_name, section, len(inter_df))


def index_dataset(
    es: Elasticsearch,
    model: SentenceTransformer,
    dataset_name: str,
    dataset_dir: Path,
    rebuild: bool,
) -> None:
    log.info(f"\n{'=' * 60}")
    log.info(f"Processing dataset: {dataset_name}")
    t0 = time.time()

    items_parquet = dataset_dir / "items.parquet"
    interactions_parquet = dataset_dir / "interactions.parquet"
    items_index = f"{INDEX_PREFIX}_{dataset_name}_items"
    inter_index = f"{INDEX_PREFIX}_{dataset_name}_interactions"

    if items_parquet.exists():
        index_items(es, model, dataset_name, items_parquet, items_index, rebuild)
    else:
        log.warning(f"  items.parquet not found for {dataset_name}, skipping items index")

    if interactions_parquet.exists():
        index_interactions(es, dataset_name, interactions_parquet, inter_index, rebuild)
    else:
        log.warning(f"  interactions.parquet not found for {dataset_name}")

    log.info(f"  Done in {time.time() - t0:.1f}s")


# Main
def main() -> None:
    parser = argparse.ArgumentParser(description="Embed items and index into Elasticsearch.")
    parser.add_argument("--dataset", default=None, help="Only index a specific dataset.")
    parser.add_argument("--rebuild", action="store_true", help="Drop and recreate indices.")
    parser.add_argument(
        "--es-host",
        default=ES_HOST,
        help=f"Elasticsearch URL (default: {ES_HOST})",
    )
    parser.add_argument(
        "--model",
        default=EMBED_MODEL,
        help=f"Sentence-transformers model (default: {EMBED_MODEL})",
    )
    args = parser.parse_args()

    try:
        es = get_es_client(args.es_host)
    except Exception as e:
        log.error(f"Cannot connect to Elasticsearch at {args.es_host}: {e}")
        log.error("Make sure ES is running: docker compose up -d")
        sys.exit(1)

    log.info(f"Loading sentence-transformer model: {args.model} ...")
    model = SentenceTransformer(args.model)
    log.info("Model loaded.")

    if args.dataset:
        dirs = [PROCESSED_DIR / args.dataset]
    else:
        dirs = sorted(PROCESSED_DIR.glob("*"))

    try:
        for dataset_dir in dirs:
            if not dataset_dir.is_dir():
                continue
            index_dataset(es, model, dataset_dir.name, dataset_dir, args.rebuild)
    except KeyboardInterrupt:
        log.warning("Interrupted by user. Progress checkpoints were saved.")
        sys.exit(130)

    log.info("\nAll datasets indexed. Visit Kibana at http://localhost:5601 to explore.")


if __name__ == "__main__":
    main()
