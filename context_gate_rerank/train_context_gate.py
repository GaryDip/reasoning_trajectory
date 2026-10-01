#!/usr/bin/env python3
"""
"Context-conditioned gate" (see conversation): instead of combined_gate_rerank's fixed
50/50 split (`final = emb - 0.25*gate_v2_score - 0.25*probe_score`, the best-so-far
combination of gate_v2/Delta and h_j-probe), learn a small per-CANDIDATE gate that decides
how to split the SAME total anomaly weight between gate_v2 and the probe:

  alpha_i   = sigmoid(w^T cand_h_pca_i + b)          -- one alpha PER CANDIDATE, not per hop
  anomaly_i = alpha_i * gate_v2_score_i + (1 - alpha_i) * probe_score_i
  final_i   = emb_score_i - LAMBDA_TOTAL * anomaly_i  -- LAMBDA_TOTAL=0.5 by default, same
                                                          total budget as the 0.25+0.25 sum
                                                          formula, so this is an apples-to-
                                                          apples comparison: same overall
                                                          anomaly weight, learned split
                                                          instead of a fixed 50/50 one.

cand_h_pca is each candidate's OWN post-hoc h_after (this hop's prefix + THIS candidate's
evidence, PCA-projected via the h_j-probe's own PCA, layer 31) — added to the lambda_data
npz by adaptive_lambda_rerank/add_probe_h_vector.py. Because it differs per candidate
(different evidence text appended), alpha can genuinely differ within the same pool — this
is the structural fix for why the EARLIER adaptive-lambda gate (conditioned on
q_main_emb/h_prev/expanded_q_emb, identical for every candidate in a pool) collapsed: that
one had no candidate-level signal to learn from. This one does.

Data comes entirely from adaptive_lambda_rerank/data/musique_{split}_lambda_data.npz
(read-only import of that project's own pool-range helper) — no new GPU extraction needed
beyond what add_probe_h_vector.py already did.

Usage:
  python train_context_gate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
LAMBDA_DIR = PROJECT_ROOT / "adaptive_lambda_rerank"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(LAMBDA_DIR))

from run_retrieval_exp import MetricAccum  # noqa: E402

from train_lambda_model import build_pool_ranges, group_split  # noqa: E402


class ContextGateDataset:
    def __init__(self, npz_path: Path):
        d = np.load(npz_path, allow_pickle=True)
        if "cand_h_pca" not in d:
            sys.exit(f"{npz_path} has no cand_h_pca — run add_probe_h_vector.py first.")
        self.K = d["K"]
        self.gold_pos = d["gold_pos"]
        self.source_id = d["source_id"]
        self.cand_emb_score = d["cand_emb_score"].astype(np.float32)
        self.cand_gate_v2_score = d["cand_gate_v2_score"].astype(np.float32)
        self.cand_abnormal_score = d["cand_abnormal_score"].astype(np.float32)
        self.cand_h_pca = d["cand_h_pca"].astype(np.float32)
        self.ranges = build_pool_ranges(d["cand_pool_idx"], len(self.K))
        self.usable = np.where(self.gold_pos >= 0)[0]

    def __len__(self) -> int:
        return len(self.usable)


class ContextGate(nn.Module):
    """alpha = sigmoid(w^T h + b) — one linear layer, ~65 params. Deliberately this small:
    see conversation on why an over-parameterized gate risks overfitting the smallest
    pooled-j bucket (j=3, K=4 only, ~4k examples in the gate-training data)."""

    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(h)).squeeze(-1)


def compute_loss(
    ds: ContextGateDataset, pool_idxs: np.ndarray, model: ContextGate, device: str, lambda_total: float,
) -> torch.Tensor:
    # One batched forward pass over every candidate in this batch of pools (alpha is a
    # per-candidate function, not per-pool, so this is more efficient than looping the model
    # call itself — only the loss aggregation loops per pool).
    slices = []
    h_chunks = []
    offset = 0
    for pool_idx in pool_idxs:
        s, e = ds.ranges[pool_idx]
        h_chunks.append(ds.cand_h_pca[s:e])
        slices.append((offset, offset + (e - s), s, e, pool_idx))
        offset += e - s
    h_cat = torch.tensor(np.concatenate(h_chunks, axis=0), device=device)
    alpha_cat = model(h_cat)

    losses = []
    for o0, o1, s, e, pool_idx in slices:
        alpha = alpha_cat[o0:o1]
        gate2 = torch.tensor(ds.cand_gate_v2_score[s:e], device=device)
        probe = torch.tensor(ds.cand_abnormal_score[s:e], device=device)
        emb = torch.tensor(ds.cand_emb_score[s:e], device=device)
        anomaly = alpha * gate2 + (1 - alpha) * probe
        final = emb - lambda_total * anomaly
        target = torch.tensor([int(ds.gold_pos[pool_idx])], device=device)
        losses.append(F.cross_entropy(final.unsqueeze(0), target))
    return torch.stack(losses).mean()


def gold_rank_of(final_scores: np.ndarray, gold_pos: int) -> int:
    order = np.argsort(-final_scores)
    return int(np.where(order == gold_pos)[0][0]) + 1


def report_fixed(ds: ContextGateDataset, pool_idxs: np.ndarray, *, lambda_gate2: float, lambda_probe: float,
                  label: str) -> None:
    accum = MetricAccum()
    for pool_idx in pool_idxs:
        s, e = ds.ranges[pool_idx]
        final = (
            ds.cand_emb_score[s:e]
            - lambda_gate2 * ds.cand_gate_v2_score[s:e]
            - lambda_probe * ds.cand_abnormal_score[s:e]
        )
        accum.update(gold_rank_of(final, int(ds.gold_pos[pool_idx])))
    print(f"[{label}] {accum.result()}")


def report_learned(ds: ContextGateDataset, pool_idxs: np.ndarray, *, alphas_by_pool: dict[int, np.ndarray],
                    lambda_total: float, label: str) -> None:
    accum = MetricAccum()
    for pool_idx in pool_idxs:
        s, e = ds.ranges[pool_idx]
        alpha = alphas_by_pool[pool_idx]
        anomaly = alpha * ds.cand_gate_v2_score[s:e] + (1 - alpha) * ds.cand_abnormal_score[s:e]
        final = ds.cand_emb_score[s:e] - lambda_total * anomaly
        accum.update(gold_rank_of(final, int(ds.gold_pos[pool_idx])))
    print(f"[{label}] {accum.result()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-data", type=Path, default=LAMBDA_DIR / "data" / "musique_train_lambda_data.npz")
    ap.add_argument("--dev-data", type=Path, default=LAMBDA_DIR / "data" / "musique_dev_lambda_data.npz")
    ap.add_argument("--out-dir", type=Path, default=HERE / "context_gate_artifacts")
    ap.add_argument("--lambda-total", type=float, default=0.5,
                     help="Same total anomaly weight as combined_gate_rerank's already-"
                          "validated sum formula (0.25 gate_v2 + 0.25 probe = 0.5 total) — "
                          "keeps this an apples-to-apples test of 'learned split' vs 'fixed "
                          "50/50 split', not a different overall lambda budget.")
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", help="Tiny model (~65 params) — CPU is plenty.")
    args = ap.parse_args()

    if not args.train_data.is_file():
        sys.exit(f"Missing {args.train_data}")
    if not args.dev_data.is_file():
        sys.exit(f"Missing {args.dev_data}")

    torch.manual_seed(args.seed)
    device = args.device

    train_ds = ContextGateDataset(args.train_data)
    dev_ds = ContextGateDataset(args.dev_data)
    print(f"Train: {len(train_ds)} usable pools (gold present) / {len(train_ds.K)} total")
    print(f"Dev:   {len(dev_ds)} usable pools (gold present) / {len(dev_ds.K)} total")

    fit_idx, holdout_idx = group_split(
        train_ds.source_id, train_ds.usable, holdout_frac=args.holdout_frac, seed=args.seed,
    )
    print(f"Fit n={len(fit_idx)}, holdout n={len(holdout_idx)} (early stopping only, dev untouched)")

    model = ContextGate(in_dim=train_ds.cand_h_pca.shape[1]).to(device)
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
            loss = compute_loss(train_ds, batch, model, device, args.lambda_total)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            holdout_loss = compute_loss(train_ds, holdout_idx, model, device, args.lambda_total).item()
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
        dev_alpha_flat = model(torch.tensor(dev_ds.cand_h_pca, device=device)).cpu().numpy()
    print(f"\nDev learned alpha (per candidate, all pools): mean={dev_alpha_flat.mean():.4f} "
          f"std={dev_alpha_flat.std():.4f} min={dev_alpha_flat.min():.4f} max={dev_alpha_flat.max():.4f}")

    alphas_by_pool = {}
    for pool_idx in dev_ds.usable:
        s, e = dev_ds.ranges[pool_idx]
        alphas_by_pool[pool_idx] = dev_alpha_flat[s:e]

    print("\nGold rank on DEV, never touched during fitting:")
    report_fixed(dev_ds, dev_ds.usable, lambda_gate2=0.0, lambda_probe=0.0, label="emb_score only")
    report_fixed(dev_ds, dev_ds.usable, lambda_gate2=0.25, lambda_probe=0.0, label="gate_v2 only (0.25)")
    report_fixed(dev_ds, dev_ds.usable, lambda_gate2=0.0, lambda_probe=0.25, label="probe only (0.25)")
    report_fixed(dev_ds, dev_ds.usable, lambda_gate2=0.25, lambda_probe=0.25,
                 label="sum (fixed 0.25/0.25, best-so-far)")
    report_learned(dev_ds, dev_ds.usable, alphas_by_pool=alphas_by_pool, lambda_total=args.lambda_total,
                   label="learned context gate")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out_dir / "context_gate.pt")
    meta = {
        "lambda_total": args.lambda_total,
        "in_dim": train_ds.cand_h_pca.shape[1],
        "n_train_pools": len(train_ds),
        "n_dev_pools": len(dev_ds),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved context_gate.pt/meta.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
