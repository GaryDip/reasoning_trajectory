#!/usr/bin/env python3
"""
Step 2: train the per-hop gate model (see model.py) on the data build_training_data.py
produced, then report whether it actually improves gold's rank within each hop's real
candidate pool compared to two frozen baselines:
  - emb_score alone (lambda=0, i.e. plain BGE ranking)
  - production's fixed lambda (default 0.25, same as --lambda-lr elsewhere)
against the LEARNED per-hop gate's ranking, using find_gold_rank + MetricAccum (same
helpers production/other scripts in this repo use) — held out on dev, never touched
during fitting.

Scoring formula for the learned gate (see model.py's docstring for why): within each
pool, z-score emb_score and -abnormal_score separately (puts both on a comparable scale,
gate=0.5 then genuinely means "trust both equally"), then take the convex combination
`gate * emb_z + (1 - gate) * abn_z`. The two frozen baselines keep using production's own
raw `emb_score - lambda * abnormal_score` formula, unchanged, so the comparison is against
the real deployed formula, not a strawman.

Usage:
  python train_lambda_model.py --train-data data/musique_train_lambda_data.npz \
      --dev-data data/musique_dev_lambda_data.npz --out-dir lambda_artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(HERE))

from run_retrieval_exp import MetricAccum  # noqa: E402

from model import LambdaModel  # noqa: E402

ZSCORE_EPS = 1e-6


def zscore(x: np.ndarray) -> np.ndarray:
    """Per-pool z-score with an epsilon guard for degenerate (std=0) pools — e.g. a pool
    where every candidate happens to get the same abnormal_score."""
    return (x - x.mean()) / (x.std() + ZSCORE_EPS)


def build_pool_ranges(cand_pool_idx: np.ndarray, n_pools: int) -> list[tuple[int, int]]:
    """cand_pool_idx is written pool-by-pool in build_training_data.py, so it's already
    sorted/contiguous per pool — just scan once for (start, end) slices."""
    ranges = [(0, 0)] * n_pools
    n = len(cand_pool_idx)
    i = 0
    while i < n:
        pid = int(cand_pool_idx[i])
        j = i
        while j < n and cand_pool_idx[j] == pid:
            j += 1
        ranges[pid] = (i, j)
        i = j
    return ranges


class LambdaDataset:
    def __init__(self, npz_path: Path):
        d = np.load(npz_path, allow_pickle=True)
        self.q_main_emb = d["q_main_emb"].astype(np.float32)
        self.h_prev = d["h_prev"].astype(np.float32)
        self.expanded_q_emb = d["expanded_q_emb"].astype(np.float32)
        self.K = d["K"]
        self.j = d["j"]
        self.gold_pos = d["gold_pos"]
        self.source_id = d["source_id"]
        self.cand_emb_score = d["cand_emb_score"]
        self.cand_abnormal_score = d["cand_abnormal_score"]
        self.cand_is_gold = d["cand_is_gold"]
        self.ranges = build_pool_ranges(d["cand_pool_idx"], len(self.K))
        self.usable = np.where(self.gold_pos >= 0)[0]

    def __len__(self) -> int:
        return len(self.usable)


def group_split(source_ids: np.ndarray, usable: np.ndarray, *, holdout_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    groups = np.unique(source_ids[usable])
    rng.shuffle(groups)
    n_holdout = max(1, int(len(groups) * holdout_frac))
    holdout_groups = set(groups[:n_holdout].tolist())
    is_holdout = np.array([source_ids[i] in holdout_groups for i in usable])
    return usable[~is_holdout], usable[is_holdout]


def compute_loss(ds: LambdaDataset, pool_idxs: np.ndarray, model: LambdaModel, device: str) -> torch.Tensor:
    q_main = torch.tensor(ds.q_main_emb[pool_idxs], device=device)
    h_prev = torch.tensor(ds.h_prev[pool_idxs], device=device)
    expanded_q = torch.tensor(ds.expanded_q_emb[pool_idxs], device=device)
    gates = model(q_main, h_prev, expanded_q)  # [B], in (0, 1)

    losses = []
    for b, pool_idx in enumerate(pool_idxs):
        s, e = ds.ranges[pool_idx]
        emb_z = zscore(ds.cand_emb_score[s:e])
        abn_z = zscore(-ds.cand_abnormal_score[s:e])  # flip sign: higher = less corrupted = better
        emb_z_t = torch.tensor(emb_z, dtype=torch.float32, device=device)
        abn_z_t = torch.tensor(abn_z, dtype=torch.float32, device=device)
        scores = gates[b] * emb_z_t + (1 - gates[b]) * abn_z_t
        target = torch.tensor([int(ds.gold_pos[pool_idx])], device=device)
        losses.append(F.cross_entropy(scores.unsqueeze(0), target))
    return torch.stack(losses).mean()


def report_gold_rank_baseline(ds: LambdaDataset, pool_idxs: np.ndarray, *, fixed_lambda: float, label: str) -> None:
    """Production's own raw formula `emb_score - fixed_lambda * abnormal_score`, unchanged
    — fixed_lambda=0.0 reproduces the emb_score-only baseline."""
    accum = MetricAccum()
    for pool_idx in pool_idxs:
        s, e = ds.ranges[pool_idx]
        emb = ds.cand_emb_score[s:e]
        abn = ds.cand_abnormal_score[s:e]
        final = emb - fixed_lambda * abn
        order = np.argsort(-final)
        gold_rank = int(np.where(order == int(ds.gold_pos[pool_idx]))[0][0]) + 1
        accum.update(gold_rank)
    print(f"[{label}] {accum.result()}")


def report_gold_rank_gate(ds: LambdaDataset, pool_idxs: np.ndarray, *, gates: np.ndarray, label: str) -> None:
    """Learned gate's formula: per-pool z-scored convex combination — see module docstring."""
    accum = MetricAccum()
    for i, pool_idx in enumerate(pool_idxs):
        s, e = ds.ranges[pool_idx]
        emb_z = zscore(ds.cand_emb_score[s:e])
        abn_z = zscore(-ds.cand_abnormal_score[s:e])
        final = gates[i] * emb_z + (1 - gates[i]) * abn_z
        order = np.argsort(-final)
        gold_rank = int(np.where(order == int(ds.gold_pos[pool_idx]))[0][0]) + 1
        accum.update(gold_rank)
    print(f"[{label}] {accum.result()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-data", type=Path, default=HERE / "data" / "musique_train_lambda_data.npz")
    ap.add_argument("--dev-data", type=Path, default=HERE / "data" / "musique_dev_lambda_data.npz")
    ap.add_argument("--out-dir", type=Path, default=HERE / "lambda_artifacts")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--use-diff", action="store_true", default=True)
    ap.add_argument("--no-use-diff", dest="use_diff", action="store_false")
    ap.add_argument("--fixed-lambda-baseline", type=float, default=0.25,
                     help="Compared against as production's own --lambda-lr default.")
    ap.add_argument("--holdout-frac", type=float, default=0.1,
                     help="Fraction of TRAIN source_ids held out for early stopping (dev is never touched).")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:1",
                     help="e.g. cuda:2. Default: cuda (whatever that resolves to) if available, "
                          "else cpu — pass explicitly on a shared cluster to avoid a busy GPU 0.")
    args = ap.parse_args()

    if not args.train_data.is_file():
        sys.exit(f"Missing {args.train_data} — run build_training_data.py --split train first.")
    if not args.dev_data.is_file():
        sys.exit(f"Missing {args.dev_data} — run build_training_data.py --split dev first.")

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = LambdaDataset(args.train_data)
    dev_ds = LambdaDataset(args.dev_data)
    print(f"Train: {len(train_ds)} usable pools (gold present) / {len(train_ds.K)} total")
    print(f"Dev:   {len(dev_ds)} usable pools (gold present) / {len(dev_ds.K)} total")

    fit_idx, holdout_idx = group_split(
        train_ds.source_id, train_ds.usable, holdout_frac=args.holdout_frac, seed=args.seed,
    )
    print(f"Fit n={len(fit_idx)}, holdout n={len(holdout_idx)} (early stopping only, dev untouched)")

    model = LambdaModel(hidden=args.hidden, use_diff=args.use_diff).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state, best_loss, bad_epochs, patience = None, float("inf"), 0, 5
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = fit_idx[rng.permutation(len(fit_idx))]
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(perm), args.batch_size):
            batch = perm[start : start + args.batch_size]
            opt.zero_grad()
            loss = compute_loss(train_ds, batch, model, device)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            holdout_loss = compute_loss(train_ds, holdout_idx, model, device).item()
        print(f"  epoch {epoch:>3}: fit_loss={total_loss / n_batches:.4f}  holdout_loss={holdout_loss:.4f}")

        if holdout_loss < best_loss - 1e-5:
            best_loss, best_state, bad_epochs = holdout_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {epoch} (best holdout_loss={best_loss:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        q_main = torch.tensor(dev_ds.q_main_emb[dev_ds.usable], device=device)
        h_prev = torch.tensor(dev_ds.h_prev[dev_ds.usable], device=device)
        expanded_q = torch.tensor(dev_ds.expanded_q_emb[dev_ds.usable], device=device)
        dev_gates = model(q_main, h_prev, expanded_q).cpu().numpy()
    print(f"\nDev learned gate: mean={dev_gates.mean():.4f} std={dev_gates.std():.4f} "
          f"min={dev_gates.min():.4f} max={dev_gates.max():.4f}")

    print("\nGold rank on DEV, never touched during fitting:")
    report_gold_rank_baseline(dev_ds, dev_ds.usable, fixed_lambda=0.0, label="emb_score only (lambda=0)")
    report_gold_rank_baseline(dev_ds, dev_ds.usable, fixed_lambda=args.fixed_lambda_baseline,
                               label=f"fixed lambda={args.fixed_lambda_baseline} (production default)")
    report_gold_rank_gate(dev_ds, dev_ds.usable, gates=dev_gates, label="learned per-hop gate")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out_dir / "lambda_model.pt")
    meta = {
        "hidden": args.hidden, "use_diff": args.use_diff,
        "fixed_lambda_baseline": args.fixed_lambda_baseline,
        "n_train_pools": len(train_ds), "n_dev_pools": len(dev_ds),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved lambda_model.pt/meta.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
