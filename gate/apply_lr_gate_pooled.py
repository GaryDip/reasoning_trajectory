#!/usr/bin/env python3
"""
Apply saved pooled LR gate artifacts (score-only, no retrain).

Usage:
  python apply_lr_gate_pooled.py --split dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
HS_ROOT = PROJECT_ROOT / "hidden_states"
RESULTS_DIR = HERE / "results_pooled"
ARTIFACTS_DIR = HERE / "artifacts_pooled"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fit_lr_gate import transition_label
from fit_lr_gate_pooled import (
    aggregate_metrics,
    collect_split_pooled,
    pooled_transition_label,
)
from lr_artifacts import load_pooled_artifacts, score_pooled_delta


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply saved pooled LR gate artifacts.")
    ap.add_argument("--split", choices=("dev", "train"), default="dev")
    ap.add_argument("--hs-root", type=Path, default=HS_ROOT)
    ap.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    ap.add_argument("--out-scores", type=Path, default=None)
    ap.add_argument("--out-metrics", type=Path, default=None)
    ap.add_argument("--max-j", type=int, default=3)
    args = ap.parse_args()

    out_scores = args.out_scores or (RESULTS_DIR / f"{args.split}_lr_pooled_scores.jsonl")
    out_metrics = args.out_metrics or (RESULTS_DIR / f"{args.split}_lr_pooled_metrics.json")
    out_scores.parent.mkdir(parents=True, exist_ok=True)

    models, meta = load_pooled_artifacts(args.artifacts_dir)
    print(f"Loaded {len(models)} pooled models from {args.artifacts_dir}")

    eval_data = collect_split_pooled(args.hs_root, args.split, max_j=args.max_j)
    score_rows: list[dict] = []
    n_missing = 0

    for j, ed in sorted(eval_data.items()):
        if j not in models:
            n_missing += len(ed["X"])
            continue
        art = models[j]
        for delta, meta_row, label in zip(ed["X"], ed["meta"], ed["y"]):
            s = score_pooled_delta(delta, art)
            K = int(meta_row["K"])
            score_rows.append({
                "id": meta_row["id"],
                "split": args.split,
                "K": K,
                "j": j,
                "transition": transition_label(j, K),
                "pooled_transition": pooled_transition_label(j),
                "pooling": "semantic_j_across_K_excluding_final",
                "trace_type": meta_row["trace_type"],
                "wrong_hops": meta_row["wrong_hops"],
                "score": round(float(s["score"]), 5),
                "threshold": round(float(s["threshold"]), 5),
                "triggered": bool(s["triggered"]),
                "should_intervene": bool(label),
                "distance": round(float(s["score"]), 5),
            })

    with out_scores.open("w", encoding="utf-8") as f:
        for r in score_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Scored {len(score_rows)} rows -> {out_scores}")
    if n_missing:
        print(f"  skipped {n_missing} deltas (no artifact)")

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(
            [r["should_intervene"] for r in score_rows],
            [r["score"] for r in score_rows],
        ))
    except Exception:
        auc = 0.0

    metrics = aggregate_metrics(
        score_rows, auc=auc, target_fpr=float(meta.get("target_fpr", 0.15)),
    )
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics -> {out_metrics}")


if __name__ == "__main__":
    main()
