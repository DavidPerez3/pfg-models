"""
03_two_tower_wide_deep.py — Two-Tower + Wide & Deep (Multi-Stage)
==================================================================
Stage 1 (Retrieval): Two-Tower embedding model → top-200 candidates.
Stage 2 (Ranking):   Wide & Deep scores the candidates with additional features.

Both stages share item embeddings (joint training).

Usage:
    python src/models/03_two_tower_wide_deep.py --dataset yelp
    python src/models/03_two_tower_wide_deep.py   # all datasets
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DATASETS, MODELS_DIR,
    add_common_args, compute_metrics, encode_users_items,
    load_interactions, log, negative_sample, print_metrics,
    save_results, train_test_split_temporal,
    apply_wandb_env, maybe_init_wandb,
)

MODEL_NAME = "two_tower_wide_deep"

# ── Hyperparameters ───────────────────────────────────────────────────────────
DIM        = 32
EPOCHS     = 10
BATCH      = 512
LR_WIDE    = 1e-2
LR_DEEP    = 1e-3
DROPOUT    = 0.15
CANDIDATES = 200     # Stage 1 retrieval size
N_NEG      = 1
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model ──────────────────────────────────────────────────────────────────────
class TwoTowerWideDeep(nn.Module):
    """
    Shared item/user embeddings (Two-Tower retrieval) +
    Wide & Deep ranking head applied on a candidate set.
    """
    def __init__(self, n_users: int, n_items: int, dim: int, dropout: float):
        super().__init__()
        # --- Shared embeddings (Tower) ---
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # --- Wide part: linear on concatenated user+item indices (feature cross) ---
        self.wide = nn.Linear(2 * dim, 1, bias=True)

        # --- Deep part: MLP on embeddings ---
        self.deep = nn.Sequential(
            nn.Linear(2 * dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def _repr(self, user_idx, item_idx):
        u = self.dropout(self.user_emb(user_idx))
        v = self.dropout(self.item_emb(item_idx))
        return torch.cat([u, v], dim=-1)          # [B, 2*dim]

    def score_wide_deep(self, user_idx, item_idx):
        """Full Wide&Deep ranking score."""
        x = self._repr(user_idx, item_idx)
        return (self.wide(x) + self.deep(x)).squeeze(-1)

    def score_tower(self, user_idx, item_idx):
        """Dot-product retrieval score (Two-Tower)."""
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        return (u * v).sum(dim=-1)


def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9).mean()


def train_model(
    model,
    train_df,
    test_df,
    n_items,
    device,
    eval_fn,
    item_embeddings_fn,
    *,
    epochs: int,
    batch_size: int,
    lr_wide: float,
    lr_deep: float,
    n_neg: int,
    wandb_run=None,
):
    df_neg = negative_sample(train_df, n_items, n_neg=n_neg)
    users     = torch.tensor(df_neg["user_idx"].values, dtype=torch.long)
    pos_items = torch.tensor(df_neg["item_idx"].values, dtype=torch.long)
    neg_items = torch.tensor(df_neg["neg_item_idx"].values, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(users, pos_items, neg_items),
        batch_size=batch_size, shuffle=True,
    )

    # Separate learning rates for wide vs. deep as per W&D paper
    wide_params = list(model.wide.parameters())
    deep_params = (
        list(model.user_emb.parameters()) +
        list(model.item_emb.parameters()) +
        list(model.deep.parameters())
    )
    optimizer = torch.optim.Adam([
        {"params": wide_params, "lr": lr_wide},
        {"params": deep_params, "lr": lr_deep},
    ])

    model.train()
    epoch_stats: list[dict[str, float]] = []
    interactive_progress = sys.stderr.isatty() or os.getenv("FORCE_TQDM", "0") == "1"
    for epoch in range(epochs):
        total = 0.0
        total_tower = 0.0
        total_wd = 0.0
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False,
            disable=not interactive_progress,
            dynamic_ncols=True,
        )
        for batch_idx, (u, p, n) in enumerate(pbar, start=1):
            u, p, n = u.to(device), p.to(device), n.to(device)
            # Stage 1 loss: retrieval (BPR on tower scores)
            loss_tower = bpr_loss(model.score_tower(u, p), model.score_tower(u, n))
            # Stage 2 loss: ranking (BPR on wide&deep scores)
            loss_wd    = bpr_loss(model.score_wide_deep(u, p), model.score_wide_deep(u, n))
            loss = 0.5 * loss_tower + 0.5 * loss_wd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            total_tower += loss_tower.item()
            total_wd += loss_wd.item()
            if interactive_progress and batch_idx % 100 == 0:
                pbar.set_postfix({"loss": f"{total / batch_idx:.4f}"})
            if (not interactive_progress) and (batch_idx % 100 == 0 or batch_idx == len(loader)):
                log.info(
                    f"      epoch {epoch+1}/{epochs} batch {batch_idx}/{len(loader)} "
                    f"loss={total / batch_idx:.4f} tower={total_tower / batch_idx:.4f} wd={total_wd / batch_idx:.4f}"
                )
        pbar.close()
        mean_total = total / max(len(loader), 1)
        mean_tower = total_tower / max(len(loader), 1)
        mean_wd = total_wd / max(len(loader), 1)
        
        # Evaluate every epoch
        log.info(f"    Epoch {epoch+1}/{epochs}  loss={mean_total:.4f} → Evaluating…")
        model.eval()
        with torch.no_grad():
            all_item_embs = item_embeddings_fn()
        metrics = eval_fn(all_item_embs)
        metrics["train/loss"] = mean_total
        metrics["train/loss_tower"] = mean_tower
        metrics["train/loss_wide_deep"] = mean_wd
        epoch_stats.append(metrics)
        
        if wandb_run is not None:
            wandb_run.log(metrics, step=epoch + 1)
        
        log.info(f"    Epoch {epoch+1}/{epochs}  NDCG@10={metrics.get('ndcg@10', 0):.4f}, Recall@10={metrics.get('recall@10', 0):.4f}")
        model.train()

    return epoch_stats


def run(dataset: str, sample: int | None, hparams: dict | None = None):
    log.info(f"\n{'='*60}\n[TT+WD] Dataset: {dataset}\n{'='*60}")
    t_start = time.time()

    hp = hparams or {
        "dim": DIM,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr_wide": LR_WIDE,
        "lr_deep": LR_DEEP,
        "dropout": DROPOUT,
        "candidates": CANDIDATES,
        "n_neg": N_NEG,
        "device": DEVICE,
    }

    df = load_interactions(dataset, sample=sample)
    df, user_map, item_map = encode_users_items(df)
    n_users, n_items = len(user_map), len(item_map)
    train_df, test_df = train_test_split_temporal(df)

    wandb_run = maybe_init_wandb(
        model_name=MODEL_NAME,
        dataset_name=dataset,
        sample=sample,
        hparams={**hp, "n_users": n_users, "n_items": n_items},
    )

    model = TwoTowerWideDeep(n_users, n_items, int(hp["dim"]), float(hp["dropout"])).to(hp["device"])
    
    # Build user-item training set for filtering seen items
    user_seen = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    
    # Define evaluation callback
    def eval_fn(all_item_embs):
        def get_recs(user_idx: int, k: int = 10):
            model.eval()
            with torch.no_grad():
                # Stage 1: Two-Tower retrieval → top CANDIDATES
                u_t = torch.tensor([user_idx], dtype=torch.long, device=hp["device"])
                u_emb = model.user_emb(u_t).detach().cpu().numpy()[0]
                tower_scores = all_item_embs @ u_emb
                seen = user_seen.get(user_idx, set())
                tower_scores[list(seen)] = -1e9
                candidate_ids = np.argsort(-tower_scores)[: int(hp["candidates"])].tolist()

                # Stage 2: Wide&Deep ranking on candidates
                u_t = torch.tensor([user_idx] * len(candidate_ids), dtype=torch.long, device=hp["device"])
                c_t = torch.tensor(candidate_ids, dtype=torch.long, device=hp["device"])
                wd_scores = model.score_wide_deep(u_t, c_t).detach().cpu().numpy()

            ranked = [candidate_ids[i] for i in np.argsort(-wd_scores)]
            return ranked[:k]
        
        return compute_metrics(
            get_recommendations=get_recs,
            test_df=test_df,
            item_embeddings=all_item_embs,
            catalog_size=n_items,
        )
    
    # Define item embeddings callback
    def item_embeddings_fn():
        model.eval()
        with torch.no_grad():
            return model.item_emb.weight.detach().cpu().numpy()
    
    log.info(
        f"  Training Two-Tower + Wide&Deep (dim={hp['dim']}, epochs={hp['epochs']}, "
        f"candidates={hp['candidates']}, n_neg={hp['n_neg']}) …"
    )
    epoch_stats = train_model(
        model,
        train_df,
        test_df,
        n_items,
        hp["device"],
        eval_fn,
        item_embeddings_fn,
        epochs=int(hp["epochs"]),
        batch_size=int(hp["batch"]),
        lr_wide=float(hp["lr_wide"]),
        lr_deep=float(hp["lr_deep"]),
        n_neg=int(hp["n_neg"]),
        wandb_run=wandb_run,
    )

    # Pre-compute final item embeddings
    model.eval()
    with torch.no_grad():
        all_item_embs = model.item_emb.weight.detach().cpu().numpy()

    def get_recs_final(user_idx: int, k: int = 10):
        model.eval()
        with torch.no_grad():
            # Stage 1: Two-Tower retrieval → top CANDIDATES
            u_t = torch.tensor([user_idx], dtype=torch.long, device=hp["device"])
            u_emb = model.user_emb(u_t).detach().cpu().numpy()[0]
            tower_scores = all_item_embs @ u_emb
            seen = user_seen.get(user_idx, set())
            tower_scores[list(seen)] = -1e9
            candidate_ids = np.argsort(-tower_scores)[: int(hp["candidates"])].tolist()

            # Stage 2: Wide&Deep ranking on candidates
            u_t = torch.tensor([user_idx] * len(candidate_ids), dtype=torch.long, device=hp["device"])
            c_t = torch.tensor(candidate_ids, dtype=torch.long, device=hp["device"])
            wd_scores = model.score_wide_deep(u_t, c_t).detach().cpu().numpy()

        ranked = [candidate_ids[i] for i in np.argsort(-wd_scores)]
        return ranked[:k]

    metrics = compute_metrics(
        get_recommendations=get_recs_final,
        test_df=test_df,
        item_embeddings=all_item_embs,
        catalog_size=n_items,
    )
    metrics["train_time_s"] = round(time.time() - t_start, 1)
    print_metrics(metrics, MODEL_NAME, dataset)

    if wandb_run is not None:
        wandb_run.finish()

    out_dir = MODELS_DIR / MODEL_NAME / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "weights.pt")
    np.save(out_dir / "item_embeddings.npy", all_item_embs)
    save_results(metrics, MODEL_NAME, dataset, hparams=hp, wandb_run=wandb_run)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Two-Tower + Wide&Deep (multi-stage)")
    add_common_args(parser)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr-wide", dest="lr_wide", type=float, default=LR_WIDE)
    parser.add_argument("--lr-deep", dest="lr_deep", type=float, default=LR_DEEP)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--candidates", type=int, default=CANDIDATES)
    parser.add_argument("--n-neg", dest="n_neg", type=int, default=N_NEG, help="Negatives per positive (default: 1).")
    parser.add_argument("--device", type=str, default=DEVICE, help="cpu, cuda, cuda:0, etc.")
    args = parser.parse_args()
    apply_wandb_env(args)
    sample = None if args.full else args.sample
    targets = [args.dataset] if args.dataset else DATASETS
    for ds in targets:
        try:
            hp = {
                "dim": args.dim,
                "epochs": args.epochs,
                "batch": args.batch,
                "lr_wide": args.lr_wide,
                "lr_deep": args.lr_deep,
                "dropout": args.dropout,
                "candidates": args.candidates,
                "n_neg": args.n_neg,
                "device": args.device,
            }
            run(ds, sample, hparams=hp)
        except Exception as e:
            log.error(f"[TT+WD] Failed on {ds}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
