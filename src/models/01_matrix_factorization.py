"""
01_matrix_factorization.py — ALS Matrix Factorization
======================================================
Uses the `implicit` library (fast CPU/GPU ALS) to learn user/item
latent factors from all 5 preprocessed datasets.

Usage:
    python src/models/01_matrix_factorization.py --dataset movielens
    python src/models/01_matrix_factorization.py            # all datasets
    python src/models/01_matrix_factorization.py --full     # no sampling
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DATASETS, MODELS_DIR,
    add_common_args, build_user_item_matrix, compute_metrics,
    encode_users_items, load_interactions, log,
    apply_wandb_env, maybe_init_wandb, print_metrics, save_results, train_test_split_temporal,
)

MODEL_NAME = "matrix_factorization"

# ── Hyperparameters ───────────────────────────────────────────────────────────
FACTORS      = 64        # latent dimensions
ITERATIONS   = 20        # ALS iterations
REGULARIZATION = 0.01
ALPHA        = 40.0      # confidence weight: c_ui = 1 + alpha * r_ui
USE_GPU      = True


def run(dataset: str, sample: int | None, hparams: dict | None = None):
    log.info(f"\n{'='*60}\n[MF] Dataset: {dataset}\n{'='*60}")
    t_start = time.time()

    hp = hparams or {
        "factors": FACTORS,
        "iterations": ITERATIONS,
        "regularization": REGULARIZATION,
        "alpha": ALPHA,
        "use_gpu": USE_GPU,
    }

    # 1. Load + encode
    df = load_interactions(dataset, sample=sample)
    df, user_map, item_map = encode_users_items(df)
    n_users = len(user_map)
    n_items = len(item_map)
    log.info(f"  Users: {n_users:,}  Items: {n_items:,}")

    wandb_run = maybe_init_wandb(
        model_name=MODEL_NAME,
        dataset_name=dataset,
        sample=sample,
        hparams={**hp, "n_users": n_users, "n_items": n_items},
    )

    # 2. Temporal split
    train_df, test_df = train_test_split_temporal(df)
    log.info(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")

    # 3. Build sparse matrix (confidence-weighted)
    train_mat = build_user_item_matrix(train_df)
    # Apply confidence weighting: C_ui = 1 + alpha * r_ui
    train_mat_conf = train_mat.copy()
    train_mat_conf.data = 1.0 + float(hp["alpha"]) * train_mat_conf.data

    # 4. Train ALS
    try:
        import implicit
    except ImportError:
        log.error("Install implicit:  pip install implicit")
        if wandb_run is not None:
            wandb_run.finish()
        return

    log.info(f"  Training ALS (factors={hp['factors']}, iters={hp['iterations']}) …")
    model = implicit.als.AlternatingLeastSquares(
        factors=int(hp["factors"]),
        iterations=int(hp["iterations"]),
        regularization=float(hp["regularization"]),
        use_gpu=bool(hp["use_gpu"]),
        random_state=42,
    )
    model.fit(train_mat_conf, show_progress=True)

    # Item embeddings for ILD
    item_embeddings = model.item_factors            # [n_items, factors]

    # 5. Recommendation function
    def get_recs(user_idx: int, k: int = 10):
        ids, scores = model.recommend(
            user_idx, train_mat[user_idx], N=k, filter_already_liked_items=True
        )
        return ids.tolist()

    # 6. Evaluate
    log.info("  Evaluating …")
    metrics = compute_metrics(
        get_recommendations=get_recs,
        test_df=test_df,
        item_embeddings=item_embeddings,
        catalog_size=n_items,
    )
    metrics["train_time_s"] = round(time.time() - t_start, 1)
    print_metrics(metrics, MODEL_NAME, dataset)

    if wandb_run is not None:
        wandb_run.log(metrics)
        wandb_run.finish()

    # 7. Save model artifacts
    out_dir = MODELS_DIR / MODEL_NAME / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "user_factors.npy", model.user_factors)
    np.save(out_dir / "item_factors.npy", model.item_factors)
    log.info(f"  Factors saved → {out_dir}")

    save_results(metrics, MODEL_NAME, dataset, hparams=hp, wandb_run=wandb_run)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Matrix Factorization (ALS)")
    add_common_args(parser)
    parser.add_argument("--factors", type=int, default=FACTORS)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--regularization", type=float, default=REGULARIZATION)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=USE_GPU)
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    args = parser.parse_args()
    apply_wandb_env(args)

    sample = None if args.full else args.sample
    targets = [args.dataset] if args.dataset else DATASETS

    for ds in targets:
        try:
            hp = {
                "factors": args.factors,
                "iterations": args.iterations,
                "regularization": args.regularization,
                "alpha": args.alpha,
                "use_gpu": args.use_gpu,
            }
            run(ds, sample, hparams=hp)
        except Exception as e:
            log.error(f"[MF] Failed on {ds}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
