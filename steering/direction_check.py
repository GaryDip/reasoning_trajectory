#!/usr/bin/env python3
"""
Step 1 of the steering exploration (see update_doc/steering_research_report.md section 3.5):
cheapest possible check of whether a gate's learned direction is a usable steering vector --
no generation, no vLLM, no GPU. Take dev-split hidden states from known negative
(counterfactual, should_intervene=True) rows, add the gate's own direction vector at
increasing strength, and see whether the gate's OWN score moves back toward "normal" --
using positive (gold, should_intervene=False) rows as a control that should NOT move much.

Direction for a single-PCA-branch gate (h-only or gate v2's Delta-only) is a straight
reprojection of the trained LogisticRegression's coefficients back through the PCA:

    w_full = lr.coef_[0] @ pca.components_        # (4096,), points toward "should intervene"
    direction = -w_full / ||w_full||               # points toward "normal"

Reuses (imports, does not copy) gate/gate_h_only/fit_lr_gate_pooled_h_only.py's
collect_split_pooled -- same data loading gate h-only's own training used, so this is
scored against real dev hidden states, not synthetic ones.

Usage:
  python steering_direction_check.py
  python steering_direction_check.py --gate h_only --alphas 0,1,2,4,8,16,32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
GATE_H_ONLY_DIR = PROJECT_ROOT / "gate" / "gate_h_only"
sys.path.insert(0, str(GATE_H_ONLY_DIR))

from fit_lr_gate_pooled_h_only import (  # noqa: E402
    DEFAULT_PER_J_LAYERS,
    HS_ROOT,
    collect_split_pooled,
    pooled_transition_label,
)


def load_h_only_artifacts(artifacts_dir: Path) -> dict[int, dict]:
    import joblib
    import json
    meta = json.loads((artifacts_dir / "meta.json").read_text(encoding="utf-8"))
    models: dict[int, dict] = {}
    for row in meta["models"]:
        models[int(row["j"])] = joblib.load(artifacts_dir / row["file"])
    return models


def direction_from_artifact(art: dict) -> np.ndarray:
    """direction points toward 'normal' -- adding it should push the score DOWN."""
    pca = art["pca_h"]
    lr = art["lr"]
    w_full = lr.coef_[0] @ pca.components_
    return -w_full / np.linalg.norm(w_full)


def score(X: np.ndarray, art: dict) -> np.ndarray:
    Z = art["pca_h"].transform(X.astype(np.float64))
    return art["lr"].predict_proba(Z)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", type=Path, default=GATE_H_ONLY_DIR / "artifacts_pooled_h_only")
    ap.add_argument("--hs-root", type=Path, default=HS_ROOT)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-j", type=int, default=3)
    ap.add_argument("--alphas", default="0,0.5,1,2,4,8,16,32,64")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "direction_check_h_only.json")
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    artifacts = load_h_only_artifacts(args.artifacts_dir)
    print(f"Loaded h-only artifacts: {args.artifacts_dir} (j -> layer: "
          f"{ {j: a['layer'] for j, a in artifacts.items()} })")

    print(f"\nLoading {args.split} split hidden states ...")
    pooled = collect_split_pooled(args.hs_root, args.split, max_j=args.max_j, layer_by_j=DEFAULT_PER_J_LAYERS)

    results: dict[str, dict] = {}
    for j in sorted(pooled):
        if j not in artifacts:
            continue
        art = artifacts[j]
        data = pooled[j]
        X = np.stack(data["X_h"], axis=0).astype(np.float64)
        y = np.array(data["y"], dtype=int)
        direction = direction_from_artifact(art)
        typical_norm = float(np.linalg.norm(X, axis=1).mean())

        X_neg = X[y == 1]  # should_intervene=True -- gate SHOULD currently flag these
        X_pos = X[y == 0]  # should_intervene=False -- gate should stay quiet on these

        print(f"\n=== j={j} ({pooled_transition_label(j)}) layer={art['layer']} "
              f"n_neg={len(X_neg)} n_pos={len(X_pos)} "
              f"mean||h||={typical_norm:.1f} ||direction||=1.0 (unit vector) ===")
        print(f"{'alpha':>8} {'alpha/||h||':>12} {'mean score (neg)':>18} "
              f"{'trig.rate (neg)':>16} {'mean score (pos)':>18} {'trig.rate (pos)':>16}")
        tau = float(art["threshold"])
        rows = []
        for alpha in alphas:
            s_neg = score(X_neg + alpha * direction, art)
            s_pos = score(X_pos + alpha * direction, art)
            row = {
                "alpha": alpha,
                "alpha_over_mean_norm": alpha / typical_norm,
                "mean_score_neg": float(s_neg.mean()),
                "trigger_rate_neg": float((s_neg > tau).mean()),
                "mean_score_pos": float(s_pos.mean()),
                "trigger_rate_pos": float((s_pos > tau).mean()),
            }
            rows.append(row)
            print(f"{alpha:8.2f} {row['alpha_over_mean_norm']:12.4f} "
                  f"{row['mean_score_neg']:18.4f} {row['trigger_rate_neg']:16.4f} "
                  f"{row['mean_score_pos']:18.4f} {row['trigger_rate_pos']:16.4f}")
        results[str(j)] = {
            "transition": pooled_transition_label(j), "layer": art["layer"],
            "n_neg": len(X_neg), "n_pos": len(X_pos), "mean_norm": typical_norm,
            "threshold": tau, "sweep": rows,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
