"""
02_two_tower.py — Two-Tower Retrieval Model
============================================
Learns separate user and item embedding towers; scores via dot product.
Training uses BPR loss with random negative sampling.

Usage:
    python src/models/02_two_tower.py --dataset movielens
    python src/models/02_two_tower.py            # all datasets
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

MODEL_NAME = "two_tower"

# ── Hyperparameters ───────────────────────────────────────────────────────────
DIM       = 32
EPOCHS    = 10
BATCH     = 512
LR        = 1e-3
DROPOUT   = 0.1
N_NEG     = 1
DEVICE    = "cpu"


# ── Model ─────────────────────────────────────────────────────────────────────
class TwoTowerModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, dropout: float):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.dropout  = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward_user(self, user_idx):
        return self.dropout(self.user_emb(user_idx))

    def forward_item(self, item_idx):
        return self.dropout(self.item_emb(item_idx))

    def score(self, user_idx, item_idx):
        u = self.forward_user(user_idx)
        v = self.forward_item(item_idx)
        return (u * v).sum(dim=-1)


def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9).mean()


# ── Training ──────────────────────────────────────────────────────────────────
def train_two_tower(model, train_df, test_df, n_items, device, eval_fn, item_embeddings_fn, *, epochs: int, batch_size: int, lr: float, n_neg: int, wandb_run=None):
    log.info(f"    Building negative samples (n_neg={n_neg}) …")
    t_neg = time.time()
    df_neg = negative_sample(train_df, n_items, n_neg=n_neg)
    log.info(f"    Negative samples ready: {len(df_neg):,} rows in {time.time() - t_neg:.1f}s")

    users = torch.tensor(df_neg["user_idx"].values, dtype=torch.long)
    pos_items = torch.tensor(df_neg["item_idx"].values, dtype=torch.long)
    neg_items = torch.tensor(df_neg["neg_item_idx"].values, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(users, pos_items, neg_items),
        batch_size=batch_size, shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    epoch_losses: list[float] = []
    epoch_metrics_list: list[dict] = []
    interactive_progress = sys.stderr.isatty() or os.getenv("FORCE_TQDM", "0") == "1"
    
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False,
            disable=not interactive_progress,
            dynamic_ncols=True,
        )
        for batch_idx, (u, p, n) in enumerate(pbar, start=1):
            u, p, n = u.to(device), p.to(device), n.to(device)
            loss = bpr_loss(model.score(u, p), model.score(u, n))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if interactive_progress and batch_idx % 100 == 0:
                pbar.set_postfix({"loss": f"{total_loss / batch_idx:.4f}"})
            if (interactive_progress and batch_idx % 1000 == 0) or ((not interactive_progress) and (batch_idx % 100 == 0 or batch_idx == len(loader))):
                log.info(f"      epoch {epoch+1}/{epochs} batch {batch_idx}/{len(loader)} loss={total_loss / batch_idx:.4f}")
        pbar.close()

        epoch_loss = total_loss / max(len(loader), 1)
        epoch_losses.append(float(epoch_loss))
        log.info(f"    Epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f} → Evaluating…")
        
        # Evaluate every epoch and log to W&B
        model.eval()
        with torch.no_grad():
            all_item_embs = item_embeddings_fn()
        metrics = eval_fn(all_item_embs)
        metrics["train/loss"] = epoch_loss
        epoch_metrics_list.append(metrics)
        
        if wandb_run is not None:
            wandb_run.log(metrics, step=epoch + 1)
        
        log.info(f"    Epoch {epoch+1} metrics: NDCG@10={metrics.get('ndcg@10', 0):.4f}, Recall@10={metrics.get('recall@10', 0):.4f}")
        model.train()

    return epoch_losses, epoch_metrics_list


def run(dataset: str, sample: int | None, hparams: dict | None = None):
    log.info(f"\n{'='*60}\n[Two-Tower] Dataset: {dataset}\n{'='*60}")
    t_start = time.time()

    hp = hparams or {
        "dim": DIM,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "dropout": DROPOUT,
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

    model = TwoTowerModel(n_users, n_items, int(hp["dim"]), float(hp["dropout"])).to(hp["device"])
    
    # Build user-item training set for filtering seen items
    user_seen = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    
    # Define evaluation callback
    def eval_fn(all_item_embs):
        def get_recs(user_idx: int, k: int = 10):
            model.eval()
            with torch.no_grad():
                u_t = torch.tensor([user_idx], dtype=torch.long, device=hp["device"])
                u_emb = model.user_emb(u_t).detach().cpu().numpy()[0]
            scores = all_item_embs @ u_emb
            seen = user_seen.get(user_idx, set())
            scores[list(seen)] = -1e9
            return np.argsort(-scores)[:k].tolist()
        
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
    
    log.info(f"  Training Two-Tower (dim={hp['dim']}, epochs={hp['epochs']}, n_neg={hp['n_neg']}) …")
    epoch_losses, epoch_metrics_list = train_two_tower(
        model,
        train_df,
        test_df,
        n_items,
        hp["device"],
        eval_fn,
        item_embeddings_fn,
        epochs=int(hp["epochs"]),
        batch_size=int(hp["batch"]),
        lr=float(hp["lr"]),
        n_neg=int(hp["n_neg"]),
        wandb_run=wandb_run,
    )

    # Pre-compute final item embeddings
    model.eval()
    with torch.no_grad():
        all_item_embs = model.item_emb.weight.detach().cpu().numpy()   # [n_items, dim]

    def get_recs_final(user_idx: int, k: int = 10):
        model.eval()
        with torch.no_grad():
            u_t = torch.tensor([user_idx], dtype=torch.long, device=hp["device"])
            u_emb = model.user_emb(u_t).detach().cpu().numpy()[0]
        scores = all_item_embs @ u_emb                  # [n_items]
        seen   = user_seen.get(user_idx, set())
        scores[list(seen)] = -1e9                        # mask seen
        return np.argsort(-scores)[:k].tolist()

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
    parser = argparse.ArgumentParser(description="Two-Tower Retrieval")
    add_common_args(parser)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
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
                "lr": args.lr,
                "dropout": args.dropout,
                "n_neg": args.n_neg,
                "device": args.device,
            }
            run(ds, sample, hparams=hp)
        except Exception as e:
            log.error(f"[Two-Tower] Failed on {ds}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
