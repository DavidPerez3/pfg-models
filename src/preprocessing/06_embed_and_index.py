"""
06_embed_and_index.py — Text Embedding + Elasticsearch Indexer
==============================================================
Reads items.parquet from each preprocessed dataset, generates text
embeddings using sentence-transformers, and indexes everything into
Elasticsearch 8.x with dense_vector fields for kNN search.

Index structure per dataset:
  pfg_<dataset_name>_items   → item catalog + text embedding
  pfg_<dataset_name>_interactions → user-item interactions (no vector)

Usage (from pfg-models/ directory):
    python src/preprocessing/06_embed_and_index.py
    python src/preprocessing/06_embed_and_index.py --dataset lastfm
    python src/preprocessing/06_embed_and_index.py --rebuild   # drop & recreate indices
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROCESSED_DIR, log

# ── Config ───────────────────────────────────────────────────────────────────
ES_HOST = "http://localhost:9200"
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, multilingual-friendly
EMBED_DIM = 384
BATCH_SIZE = 256                    # rows per embedding batch
BULK_CHUNK = 500                    # docs per ES bulk request

INDEX_PREFIX = "pfg"

# ── Dataset-specific text field builders ────────────────────────────────────
# Each function receives a row (pd.Series) and returns a plain string
# that will be embedded as the item's semantic vector.

def text_movielens(row: pd.Series) -> str:
    parts = [str(row.get("title", ""))]
    genres = row.get("genres_str", "") or ""
    if genres and genres != "(no genres listed)":
        parts.append(genres.replace("|", " "))
    return " | ".join(p for p in parts if p)

def text_amazon(row: pd.Series) -> str:
    parts = [str(row.get("item_id", ""))]
    return " ".join(p for p in parts if p)

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
    "movielens":           text_movielens,
    "amazon_movies":       text_amazon,
    "amazon_electronics":  text_amazon,
    "yelp":                text_yelp,
    "lastfm":              text_lastfm,
    "foursquare":          text_foursquare,
}

# ── Index mappings ────────────────────────────────────────────────────────────
def items_mapping() -> dict:
    return {
        "mappings": {
            "properties": {
                "item_id":    {"type": "keyword"},
                "dataset":    {"type": "keyword"},
                "text_repr":  {"type": "text"},
                "embedding":  {
                    "type":       "dense_vector",
                    "dims":       EMBED_DIM,
                    "index":      True,
                    "similarity": "cosine",
                },
                # Generic fields (populated when available)
                "title":      {"type": "text"},
                "genres":     {"type": "keyword"},
                "artist":     {"type": "keyword"},
                "track":      {"type": "keyword"},
                "categories": {"type": "keyword"},
                "city":       {"type": "keyword"},
                "avg_rating": {"type": "float"},
                "year":       {"type": "float"},
            }
        },
        "settings": {
            "number_of_shards":   1,
            "number_of_replicas": 0,
        },
    }

def interactions_mapping() -> dict:
    return {
        "mappings": {
            "properties": {
                "user_id":   {"type": "keyword"},
                "item_id":   {"type": "keyword"},
                "rating":    {"type": "float"},
                "timestamp": {"type": "date", "format": "epoch_second"},
                "dataset":   {"type": "keyword"},
                # Optional extra fields indexed as-is
                "play_count":       {"type": "integer"},
                "check_in_count":   {"type": "integer"},
                "review_length":    {"type": "integer"},
                "vote":             {"type": "integer"},
                "hour_of_day":      {"type": "byte"},
                "day_of_week":      {"type": "byte"},
                "month":            {"type": "byte"},
            }
        },
        "settings": {
            "number_of_shards":   1,
            "number_of_replicas": 0,
        },
    }

# ── Elasticsearch helpers ────────────────────────────────────────────────────
def get_es_client() -> Elasticsearch:
    es = Elasticsearch(ES_HOST, request_timeout=60)
    info = es.info()
    log.info(f"Connected to Elasticsearch {info['version']['number']} at {ES_HOST}")
    return es

def ensure_index(es: Elasticsearch, index: str, mapping: dict, rebuild: bool = False):
    if es.indices.exists(index=index):
        if rebuild:
            log.info(f"Deleting existing index: {index}")
            es.indices.delete(index=index)
        else:
            log.info(f"Index already exists (use --rebuild to recreate): {index}")
            return
    log.info(f"Creating index: {index}")
    es.indices.create(index=index, body=mapping)

def bulk_index(es: Elasticsearch, index: str, docs: list[dict]):
    actions = [{"_index": index, "_source": doc} for doc in docs]
    ok, errors = helpers.bulk(es, actions, chunk_size=BULK_CHUNK, raise_on_error=False)
    if errors:
        log.warning(f"  {len(errors)} bulk errors (first: {errors[0]})")
    return ok

# ── Per-dataset processing ────────────────────────────────────────────────────
def index_dataset(
    es: Elasticsearch,
    model: SentenceTransformer,
    dataset_name: str,
    dataset_dir: Path,
    rebuild: bool,
):
    log.info(f"\n{'='*60}")
    log.info(f"Processing dataset: {dataset_name}")
    t0 = time.time()

    items_parquet = dataset_dir / "items.parquet"
    interactions_parquet = dataset_dir / "interactions.parquet"

    items_index = f"{INDEX_PREFIX}_{dataset_name}_items"
    inter_index = f"{INDEX_PREFIX}_{dataset_name}_interactions"

    # ── Items index ──────────────────────────────────────────────────────────
    if items_parquet.exists():
        ensure_index(es, items_index, items_mapping(), rebuild=rebuild)
        items_df = pd.read_parquet(items_parquet)
        log.info(f"  Items: {len(items_df):,} rows")

        text_fn = TEXT_BUILDERS.get(dataset_name, lambda r: str(r.get("item_id", "")))

        # Process in batches
        total_indexed = 0
        for start in range(0, len(items_df), BATCH_SIZE):
            batch = items_df.iloc[start : start + BATCH_SIZE]
            texts = [text_fn(row) for _, row in batch.iterrows()]
            embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)

            docs = []
            for (_, row), text, emb in zip(batch.iterrows(), texts, embeddings):
                doc = {
                    "item_id":   str(row.get("item_id", "")),
                    "dataset":   dataset_name,
                    "text_repr": text,
                    "embedding": emb.tolist(),
                }
                # Add optional metadata fields (only if they exist and are not NaN)
                for src, dst in [
                    ("title", "title"), ("genres_str", "genres"),
                    ("artist_name", "artist"), ("track_name", "track"),
                    ("categories", "categories"), ("city", "city"),
                    ("business_avg_stars", "avg_rating"), ("year", "year"),
                ]:
                    val = row.get(src)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        doc[dst] = val
                docs.append(doc)

            indexed = bulk_index(es, items_index, docs)
            total_indexed += indexed
            log.info(f"  Indexed items {start}–{start + len(batch)}: {indexed} OK")

        log.info(f"  Total items indexed: {total_indexed:,}")
    else:
        log.warning(f"  items.parquet not found for {dataset_name}, skipping items index")

    # ── Interactions index ───────────────────────────────────────────────────
    if interactions_parquet.exists():
        ensure_index(es, inter_index, interactions_mapping(), rebuild=rebuild)
        inter_df = pd.read_parquet(interactions_parquet)
        log.info(f"  Interactions: {len(inter_df):,} rows")

        # Drop integer index columns (not needed in ES)
        cols_to_drop = [c for c in inter_df.columns if c.endswith("_idx")]
        inter_df = inter_df.drop(columns=cols_to_drop, errors="ignore")

        docs_buf = []
        total_indexed = 0
        for i, (_, row) in enumerate(inter_df.iterrows()):
            doc = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                   for k, v in row.items() if v is not None}
            # Ensure types that ES needs
            doc["user_id"] = str(doc.get("user_id", ""))
            doc["item_id"] = str(doc.get("item_id", ""))
            doc["rating"]  = float(doc.get("rating", 0))
            doc["dataset"] = dataset_name
            docs_buf.append(doc)

            if len(docs_buf) >= BULK_CHUNK:
                total_indexed += bulk_index(es, inter_index, docs_buf)
                docs_buf = []

        if docs_buf:
            total_indexed += bulk_index(es, inter_index, docs_buf)

        log.info(f"  Total interactions indexed: {total_indexed:,}")
    else:
        log.warning(f"  interactions.parquet not found for {dataset_name}")

    log.info(f"  Done in {time.time() - t0:.1f}s")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Embed items and index into Elasticsearch.")
    parser.add_argument("--dataset", default=None, help="Only index a specific dataset.")
    parser.add_argument("--rebuild", action="store_true", help="Drop and recreate indices.")
    parser.add_argument("--es-host", default=ES_HOST, help=f"Elasticsearch URL (default: {ES_HOST})")
    parser.add_argument("--model", default=EMBED_MODEL, help=f"Sentence-transformers model (default: {EMBED_MODEL})")
    args = parser.parse_args()

    # Connect to ES
    try:
        es = get_es_client()
    except Exception as e:
        log.error(f"Cannot connect to Elasticsearch at {args.es_host}: {e}")
        log.error("Make sure ES is running:  docker compose up -d")
        sys.exit(1)

    # Load embedding model
    log.info(f"Loading sentence-transformer model: {args.model} …")
    model = SentenceTransformer(args.model)
    log.info("Model loaded.")

    # Discover datasets
    if args.dataset:
        dirs = [PROCESSED_DIR / args.dataset]
    else:
        dirs = sorted(PROCESSED_DIR.glob("*"))

    for dataset_dir in dirs:
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        index_dataset(es, model, dataset_name, dataset_dir, rebuild=args.rebuild)

    log.info("\nAll datasets indexed. Visit Kibana at http://localhost:5601 to explore.")


if __name__ == "__main__":
    main()
