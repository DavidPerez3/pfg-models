"""
04_sasrec.py — SASRec: Self-Attentive Sequential Recommendation
================================================================
Transformer encoder (2-layer, causal mask) trained on per-user item
sequences ordered by timestamp. Predicts the next item via BCE loss.

Usage:
    python src/models/04_sasrec.py --dataset foursquare
    python src/models/04_sasrec.py   # all datasets
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DATASETS, MODELS_DIR, IMPLICIT_DATASETS,
    add_common_args, build_sequences, compute_metrics, encode_users_items,
    load_interactions, log, print_metrics, save_results,
    train_test_split_temporal,
    apply_wandb_env, maybe_init_wandb,
)

MODEL_NAME = "sasrec"

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_LEN   = 50
D_MODEL   = 32
N_HEADS   = 2
N_LAYERS  = 2
DROPOUT   = 0.2
EPOCHS    = 10
BATCH     = 256
LR        = 1e-3
NEG_SAMPLES = 1     # negative items per positive
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


# ── SASRec Model ──────────────────────────────────────────────────────────────
class SASRec(nn.Module):
    def __init__(self, n_items: int, d_model: int, n_heads: int,
                 n_layers: int, max_len: int, dropout: float):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)  # 0 = pad
        self.pos_emb  = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        nn.init.xavier_uniform_(self.item_emb.weight[1:])  # skip pad

    def _causal_mask(self, seq_len: int, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: [B, L] item indices (0=pad). Returns [B, L, d_model]."""
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.dropout(self.item_emb(seq) + self.pos_emb(pos))
        key_padding_mask = (seq == 0)            # True where padded
        causal = self._causal_mask(L, seq.device)
        x = self.transformer(x, mask=causal, src_key_padding_mask=key_padding_mask)
        return self.norm(x)                      # [B, L, d]

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Return encoding of the last non-pad position. [B, d]"""
        h = self.encode(seq)                     # [B, L, d]
        # Index last real (non-pad) position
        lengths = (seq != 0).sum(dim=1) - 1     # [B]
        lengths = lengths.clamp(min=0)
        last = h[torch.arange(h.size(0)), lengths]   # [B, d]
        return last


# ── Dataset ───────────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, sequences, n_items, max_len, neg_samples=1, rng=None):
        self.seqs = sequences          # list of item-idx lists (1-indexed, 0=pad)
        self.n_items   = n_items
        self.max_len   = max_len
        self.neg_k     = neg_samples
        self.rng       = rng or np.random.default_rng(42)

    def __len__(self):
        return len(self.seqs)

    def _pad(self, seq):
        seq = seq[-self.max_len:]
        pad = [0] * (self.max_len - len(seq))
        return pad + seq

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if len(seq) < 2:
            seq = seq + seq               # duplicate if single item
        input_seq  = self._pad(seq[:-1])  # all but last
        target_pos = seq[-1]
        # Random negative
        while True:
            neg = int(self.rng.integers(1, self.n_items + 1))
            if neg not in set(seq):
                break
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


# ── Training ──────────────────────────────────────────────────────────────────
def train_sasrec(
    model,
    sequences,
    seqs_dict,
    test_df,
    n_items,
    device,
    eval_fn,
    item_embeddings_fn,
    *,
    max_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    neg_samples: int,
    wandb_run=None,
):
    dataset = SeqDataset(sequences, n_items, max_len, neg_samples=neg_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.BCEWithLogitsLoss()

    model.train()
    epoch_losses: list[float] = []
    epoch_metrics_list: list[dict] = []
    interactive_progress = sys.stderr.isatty() or os.getenv("FORCE_TQDM", "0") == "1"
    
    for epoch in range(epochs):
        total = 0.0
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False,
            disable=not interactive_progress,
            dynamic_ncols=True,
        )
        for batch_idx, (seq, pos, neg) in enumerate(pbar, start=1):
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            h = model(seq)                                    # [B, d]
            pos_emb = model.item_emb(pos)                    # [B, d]
            neg_emb = model.item_emb(neg)                    # [B, d]
            pos_scores = (h * pos_emb).sum(dim=-1)           # [B]
            neg_scores = (h * neg_emb).sum(dim=-1)
            labels_pos = torch.ones_like(pos_scores)
            labels_neg = torch.zeros_like(neg_scores)
            loss = (criterion(pos_scores, labels_pos) +
                    criterion(neg_scores, labels_neg)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            if interactive_progress and batch_idx % 100 == 0:
                pbar.set_postfix({"loss": f"{total / batch_idx:.4f}"})
            if (not interactive_progress) and (batch_idx % 100 == 0 or batch_idx == len(loader)):
                log.info(f"      epoch {epoch+1}/{epochs} batch {batch_idx}/{len(loader)} loss={total / batch_idx:.4f}")
        pbar.close()
        epoch_loss = total / max(len(loader), 1)
        epoch_losses.append(float(epoch_loss))
        log.info(f"    Epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f} → Evaluating…")
        
        # Evaluate every epoch
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
    log.info(f"\n{'='*60}\n[SASRec] Dataset: {dataset}\n{'='*60}")
    t_start = time.time()

    hp = hparams or {
        "max_len": MAX_LEN,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "neg_samples": NEG_SAMPLES,
        "device": DEVICE,
    }

    df = load_interactions(dataset, sample=sample)
    # For explicit datasets, keep only high-rating interactions as positives
    if dataset not in IMPLICIT_DATASETS:
        threshold = df["rating"].quantile(0.6)
        df = df[df["rating"] >= threshold].reset_index(drop=True)

    df, user_map, item_map = encode_users_items(df)
    n_items = len(item_map)
    train_df, test_df = train_test_split_temporal(df)

    # Build per-user sequences (item_idx + 1 to leave 0 as padding)
    df_shifted = train_df.copy()
    df_shifted["item_idx"] = df_shifted["item_idx"] + 1   # 1-indexed
    seqs_dict = build_sequences(df_shifted, max_len=int(hp["max_len"]) + 1)
    sequences = [v for v in seqs_dict.values() if len(v) >= 2]

    wandb_run = maybe_init_wandb(
        model_name=MODEL_NAME,
        dataset_name=dataset,
        sample=sample,
        hparams={**hp, "n_items": n_items},
    )

    model = SASRec(
        n_items,
        int(hp["d_model"]),
        int(hp["n_heads"]),
        int(hp["n_layers"]),
        int(hp["max_len"]),
        float(hp["dropout"]),
    ).to(hp["device"])
    
    # Build per-user history for masking
    user_seqs = {u: set(s) for u, s in seqs_dict.items()}
    
    # Define evaluation callback
    def eval_fn(all_item_embs):
        def make_seq_tensor(user_idx):
            seq = seqs_dict.get(user_idx, [])[-int(hp["max_len"]):]
            padded = [0] * (int(hp["max_len"]) - len(seq)) + seq
            return torch.tensor([padded], dtype=torch.long)
        
        def get_recs(user_idx: int, k: int = 10):
            model.eval()
            with torch.no_grad():
                seq_t = make_seq_tensor(user_idx).to(hp["device"])
                h = model(seq_t)[0].detach().cpu().numpy()   # [d]
            scores = all_item_embs @ h                        # [n_items]
            seen = user_seqs.get(user_idx, set())
            seen_0idx = {s - 1 for s in seen if s > 0}
            scores[list(seen_0idx)] = -1e9
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
            return model.item_emb.weight[1:].detach().cpu().numpy()
    
    log.info(
        f"  Training SASRec (d={hp['d_model']}, heads={hp['n_heads']}, layers={hp['n_layers']}, "
        f"epochs={hp['epochs']}, neg_samples={hp['neg_samples']}) …"
    )
    epoch_losses, epoch_metrics_list = train_sasrec(
        model,
        sequences,
        seqs_dict,
        test_df,
        n_items,
        hp["device"],
        eval_fn,
        item_embeddings_fn,
        max_len=int(hp["max_len"]),
        epochs=int(hp["epochs"]),
        batch_size=int(hp["batch"]),
        lr=float(hp["lr"]),
        neg_samples=int(hp["neg_samples"]),
        wandb_run=wandb_run,
    )

    # Item embeddings for ILD (use raw embedding layer, 1-indexed → shift back)
    model.eval()
    with torch.no_grad():
        raw_embs = model.item_emb.weight[1:].detach().cpu().numpy()    # [n_items, d]

    def make_seq_tensor_final(user_idx):
        seq = seqs_dict.get(user_idx, [])[-int(hp["max_len"]):]
        padded = [0] * (int(hp["max_len"]) - len(seq)) + seq
        return torch.tensor([padded], dtype=torch.long)

    def get_recs_final(user_idx: int, k: int = 10):
        model.eval()
        with torch.no_grad():
            seq_t = make_seq_tensor_final(user_idx).to(hp["device"])
            h = model(seq_t)[0].detach().cpu().numpy()   # [d]
        scores = raw_embs @ h                            # [n_items]
        seen = user_seqs.get(user_idx, set())
        seen_0idx = {s - 1 for s in seen if s > 0}
        scores[list(seen_0idx)] = -1e9
        return np.argsort(-scores)[:k].tolist()

    # Shift test_df item_idx back (encoder output is 0-indexed)
    # (test_df uses item_idx from encode_users_items which is 0-indexed)
    metrics = compute_metrics(
        get_recommendations=get_recs_final,
        test_df=test_df,
        item_embeddings=raw_embs,
        catalog_size=n_items,
    )
    metrics["train_time_s"] = round(time.time() - t_start, 1)
    print_metrics(metrics, MODEL_NAME, dataset)

    if wandb_run is not None:
        wandb_run.finish()

    out_dir = MODELS_DIR / MODEL_NAME / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "weights.pt")
    np.save(out_dir / "item_embeddings.npy", raw_embs)
    save_results(metrics, MODEL_NAME, dataset, hparams=hp, wandb_run=wandb_run)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SASRec Sequential Recommendation")
    add_common_args(parser)
    parser.add_argument("--max-len", dest="max_len", type=int, default=MAX_LEN)
    parser.add_argument("--d-model", dest="d_model", type=int, default=D_MODEL)
    parser.add_argument("--n-heads", dest="n_heads", type=int, default=N_HEADS)
    parser.add_argument("--n-layers", dest="n_layers", type=int, default=N_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--neg-samples", dest="neg_samples", type=int, default=NEG_SAMPLES)
    parser.add_argument("--device", type=str, default=DEVICE, help="cpu, cuda, cuda:0, etc.")
    args = parser.parse_args()
    apply_wandb_env(args)
    sample = None if args.full else args.sample
    targets = [args.dataset] if args.dataset else DATASETS
    for ds in targets:
        try:
            hp = {
                "max_len": args.max_len,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
                "epochs": args.epochs,
                "batch": args.batch,
                "lr": args.lr,
                "neg_samples": args.neg_samples,
                "device": args.device,
            }
            run(ds, sample, hparams=hp)
        except Exception as e:
            log.error(f"[SASRec] Failed on {ds}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
