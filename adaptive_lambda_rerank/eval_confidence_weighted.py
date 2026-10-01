#!/usr/bin/env python3
"""
Offline validation of the "confidence-weighted" combination idea (see conversation) —
needs a musique_{split}_lambda_data.npz that already has cand_gate_v2_score added by
add_gate_v2_scores.py (on top of the cand_emb_score/cand_abnormal_score build_training_data.py
already put there). No GPU needed: everything here is pure numpy over already-extracted
per-candidate scores.

Per pool, instead of a fixed lambda for gate_v2 and the probe, weight them by how DECISIVE
each one's own scores are for THIS specific candidate pool (std across the pool's
candidates) — a pool where gate_v2's scores are spread out (confident about its ranking)
gets more gate_v2 weight; a pool where the probe's scores barely vary (no real signal this
time) gets less probe weight. No training involved, and confidence_gate == confidence_probe
exactly reproduces the fixed-0.25-each "sum" formula (already validated end-to-end).

Usage:
  python eval_confidence_weighted.py --data data/musique_dev_lambda_data.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))

from run_retrieval_exp import MetricAccum  # noqa: E402

from train_lambda_model import build_pool_ranges  # noqa: E402

EPS = 1e-6


def gold_rank_of(final_scores: np.ndarray, gold_pos: int) -> int:
    order = np.argsort(-final_scores)
    return int(np.where(order == gold_pos)[0][0]) + 1


def coarse_then_fine_rank(
    e_: np.ndarray, a_: np.ndarray, g_: np.ndarray, gp: int, *, lambda_coarse: float, lambda_fine: float, k: int,
) -> int:
    """Stage 1 (coarse): probe + similarity narrows the FULL pool down to the top-k by
    emb_score - lambda_coarse*probe_score. Stage 2 (fine): gate_v2 + similarity re-ranks
    ONLY that shortlist by emb_score - lambda_fine*gate_v2_score. If gold gets filtered out
    at stage 1, its rank is reported as its stage-1 (coarse) rank directly — which is by
    definition > k, so it still counts as a correct miss for recall@1/3, and MRR reflects
    how badly stage 1 excluded it rather than an arbitrary sentinel."""
    n = len(e_)
    k = min(k, n)
    coarse_final = e_ - lambda_coarse * a_
    coarse_order = np.argsort(-coarse_final)
    shortlist = coarse_order[:k]
    if gp not in shortlist:
        return int(np.where(coarse_order == gp)[0][0]) + 1
    fine_scores = e_[shortlist] - lambda_fine * g_[shortlist]
    fine_order = shortlist[np.argsort(-fine_scores)]
    return int(np.where(fine_order == gp)[0][0]) + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--lambda-base", type=float, default=0.25,
                     help="Base lambda each signal gets when confidences are equal — same "
                          "value each signal already uses successfully on its own.")
    ap.add_argument("--coarse-then-fine-ks", type=int, nargs="+", default=[3, 5],
                     help="Shortlist sizes to test for the probe-coarse / gate_v2-fine "
                          "two-stage pipeline.")
    args = ap.parse_args()

    if not args.data.is_file():
        sys.exit(f"Missing {args.data}")
    d = np.load(args.data, allow_pickle=True)
    if "cand_gate_v2_score" not in d:
        sys.exit(f"{args.data} has no cand_gate_v2_score — run add_gate_v2_scores.py first.")

    K = d["K"]
    gold_pos = d["gold_pos"]
    emb = d["cand_emb_score"]
    abn = d["cand_abnormal_score"]
    gate2 = d["cand_gate_v2_score"]
    ranges = build_pool_ranges(d["cand_pool_idx"], len(K))
    usable = np.where(gold_pos >= 0)[0]
    print(f"{args.data}: {len(usable)} usable pools (gold present) / {len(K)} total")

    accums = {
        "emb_score only (lambda=0)": MetricAccum(),
        "gate_v2 only (fixed 0.25)": MetricAccum(),
        "probe only (fixed 0.25)": MetricAccum(),
        "sum (gate_v2 0.25 + probe 0.25)": MetricAccum(),
        "confidence-weighted": MetricAccum(),
        **{f"coarse(probe)->fine(gate_v2) k={k}": MetricAccum() for k in args.coarse_then_fine_ks},
    }

    for pool_idx in usable:
        s, e = ranges[pool_idx]
        e_ = emb[s:e]
        a_ = abn[s:e]
        g_ = gate2[s:e]
        gp = int(gold_pos[pool_idx])

        accums["emb_score only (lambda=0)"].update(gold_rank_of(e_, gp))
        accums["gate_v2 only (fixed 0.25)"].update(gold_rank_of(e_ - args.lambda_base * g_, gp))
        accums["probe only (fixed 0.25)"].update(gold_rank_of(e_ - args.lambda_base * a_, gp))
        accums["sum (gate_v2 0.25 + probe 0.25)"].update(
            gold_rank_of(e_ - args.lambda_base * g_ - args.lambda_base * a_, gp)
        )

        conf_gate = float(g_.std())
        conf_probe = float(a_.std())
        w_gate = conf_gate / (conf_gate + conf_probe + EPS)
        w_probe = 1.0 - w_gate
        final = e_ - (2 * args.lambda_base * w_gate) * g_ - (2 * args.lambda_base * w_probe) * a_
        accums["confidence-weighted"].update(gold_rank_of(final, gp))

        for k in args.coarse_then_fine_ks:
            rank = coarse_then_fine_rank(
                e_, a_, g_, gp, lambda_coarse=args.lambda_base, lambda_fine=args.lambda_base, k=k,
            )
            accums[f"coarse(probe)->fine(gate_v2) k={k}"].update(rank)

    print()
    for label, accum in accums.items():
        print(f"[{label}] {accum.result()}")


if __name__ == "__main__":
    main()
