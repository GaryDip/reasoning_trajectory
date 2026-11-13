#!/usr/bin/env python3
"""Re-do the per-transition layer selection on a TRAIN-INTERNAL held-out split, so the choice
no longer depends on dev.

Background: the production per-j layers (j=0,1 -> 15; j=2,3 -> 23) were picked by comparing dev
AUC (gate/compare_layers_pilot.py evaluates on hidden_states/pilot_multilayer/dev/). That dev
split is the same MuSiQue dev the end-to-end tables report on, so the layer choice is a
hyperparameter tuned on the evaluation split. λ had the same problem and was re-tuned on train
in the 0831 round; this script does the equivalent for the layer choice.

Procedure, for each candidate layer L and each pooled transition j:
  1. split the TRAIN traces by question (`source_id`, so a question's gold trace and all of its
     counterfactual variants land on the same side) into fit / select
  2. fit PCA + LogisticRegression on the fit side, exactly as production does
  3. score AUC on the select side
  4. report the argmax layer per j and compare it with the production choice

Features mirror production: 'delta' reproduces the v2-era decision procedure the layers were
originally picked with; 'h_delta' is the feature gate v3 actually uses (concat of independent
PCAs). Both are reported — the layer choice should ideally agree.

All four layers are read in a single pass over the npz files (one read each, ~1 min for the full
83,927 traces), then held in RAM as float32.

Usage:
  python select_layers_train_holdout.py                       # all traces, both feature variants
  python select_layers_train_holdout.py --cap-per-bucket 4000 # faster smoke run
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gate_v3"))
sys.path.insert(0, str(HERE))

# reuse production's label / manifest / K parsing so the rows are identical to what it trains on
from fit_lr_gate_pooled_v3 import (  # noqa: E402
    load_manifest_meta,
    parse_wrong_hops,
    read_K_from_npz,
    should_intervene_at_j,
)

PRODUCTION_LAYERS = {0: 15, 1: 15, 2: 23, 3: 23}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hs-root", type=Path, default=HERE.parent / "hidden_states/pilot_multilayer")
    ap.add_argument("--split", default="train_pilot", help="directory under --hs-root (train side)")
    ap.add_argument("--layers", default="7,15,23,31")
    ap.add_argument("--max-j", type=int, default=3)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--select-ratio", type=float, default=0.2, help="fraction of QUESTIONS held out for layer选择")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", default="delta,h_delta", help="comma list of: delta, h_delta")
    ap.add_argument("--cap-per-bucket", type=int, default=0,
                     help="cap traces per (K, label) bucket for a faster run (0 = use all)")
    ap.add_argument("--out", type=Path, default=HERE / "pilot_layer_compare/layer_selection_train_holdout.json")
    return ap.parse_args()


def load_rows(hs_root: Path, split: str, layers: list[int], max_j: int, cap: int):
    """One pass over the npz files; returns per-(layer, j) feature blocks plus labels/question ids.

    rows[(layer, j)] = {"delta": [vec...], "h": [vec...]}; labels/qids are shared across layers
    because every layer sees exactly the same (trace, j) rows.
    """
    manifest_path = hs_root / split / "manifest.jsonl"
    manifest = load_manifest_meta(manifest_path)
    qid_of = {}
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                qid_of[str(r["id"]).strip()] = str(r.get("source_id") or r["id"]).strip()

    feats = {(l, j): {"delta": [], "h": []} for l in layers for j in range(max_j + 1)}
    labels = defaultdict(list)
    qids = defaultdict(list)
    bucket_count = defaultdict(int)
    n_files = n_used = 0

    for sd in ("pos", "neg"):
        npz_dir = hs_root / split / "activations" / sd
        if not npz_dir.is_dir():
            continue
        trace_type = "correct" if sd == "pos" else "error"
        for fp in sorted(npz_dir.glob("*.npz")):
            n_files += 1
            z = np.load(fp, allow_pickle=True)
            hidden = z["hidden"]                                  # (n_layers, n_prefix, hidden_dim)
            file_layers = [int(x) for x in z["layers"]]
            eid = str(np.asarray(z["example_id"]).reshape(-1)[0]).strip()
            if not eid:
                continue
            K = read_K_from_npz(z, hidden[0])
            if K < 1:
                continue
            if cap:
                key = (K, trace_type)
                if bucket_count[key] >= cap:
                    continue
                bucket_count[key] += 1
            n_used += 1
            wrong_hops = parse_wrong_hops(manifest.get(eid), z)
            qid = qid_of.get(eid, eid)
            n_prefix = hidden.shape[1]
            for j in range(min(n_prefix - 1, K, max_j + 1)):
                y = should_intervene_at_j(wrong_hops, j, trace_type=trace_type)
                labels[j].append(int(y))
                qids[j].append(qid)
                for l in layers:
                    li = file_layers.index(l)
                    h_after = hidden[li][j + 1].astype(np.float32)
                    feats[(l, j)]["delta"].append(h_after - hidden[li][j].astype(np.float32))
                    feats[(l, j)]["h"].append(h_after)
            if n_files % 20000 == 0:
                print(f"  read {n_files} npz ...", flush=True)

    print(f"  读取完成: {n_files} 个 npz，使用 {n_used} 条 trace")
    return feats, labels, qids


def fit_and_score(X_blocks: list[np.ndarray], y: np.ndarray, fit_mask: np.ndarray,
                  pca_dim: int, C: float) -> float:
    """Independent PCA per block (as production does for h and delta), then one LR. Returns
    select-side AUC."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    Z_fit, Z_sel = [], []
    for X in X_blocks:
        nc = min(pca_dim, X[fit_mask].shape[0], X.shape[1])
        pca = PCA(n_components=nc, random_state=0).fit(X[fit_mask].astype(np.float64))
        Z_fit.append(pca.transform(X[fit_mask].astype(np.float64)))
        Z_sel.append(pca.transform(X[~fit_mask].astype(np.float64)))
    Zf, Zs = np.concatenate(Z_fit, axis=1), np.concatenate(Z_sel, axis=1)
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=0, solver="lbfgs")
    lr.fit(Zf, y[fit_mask])
    proba = lr.predict_proba(Zs)[:, 1]
    y_sel = y[~fit_mask]
    if len(np.unique(y_sel)) < 2:
        return float("nan")
    return float(roc_auc_score(y_sel, proba))


def main() -> None:
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    variants = [v.strip() for v in args.features.split(",") if v.strip()]
    print(f"候选层: {layers} | 特征变体: {variants} | PCA={args.pca_dim} C={args.C}")
    print(f"数据: {args.hs_root / args.split}（按 source_id 切分，留出 {args.select_ratio:.0%} 的题目用于选层）")

    feats, labels, qids = load_rows(args.hs_root, args.split, layers, args.max_j, args.cap_per_bucket)

    rng = np.random.default_rng(args.seed)
    all_q = sorted({q for j in qids for q in qids[j]})
    sel_q = set(rng.choice(np.array(all_q), size=int(len(all_q) * args.select_ratio), replace=False).tolist())
    print(f"题目总数 {len(all_q)}，其中 {len(sel_q)} 道进入选层集\n")

    report = {"layers": layers, "production_layers": PRODUCTION_LAYERS, "select_ratio": args.select_ratio,
              "seed": args.seed, "pca_dim": args.pca_dim, "C": args.C, "per_feature": {}}

    for variant in variants:
        print(f"===== 特征 = {variant} " + ("（复现当初选层时用的 v2 口径）" if variant == "delta" else "（gate v3 实际使用的口径）"))
        header = f"{'j':>3} {'转移':<10} {'行数':>8} {'正例率':>8} " + "".join(f"{'layer '+str(l):>11}" for l in layers) + f"{'最优层':>9}{'生产层':>8}"
        print(header)
        per_j = {}
        for j in range(args.max_j + 1):
            y = np.array(labels[j], dtype=int)
            if len(y) == 0:
                continue
            q = np.array(qids[j])
            fit_mask = ~np.isin(q, list(sel_q))
            aucs = {}
            for l in layers:
                blocks = [np.asarray(feats[(l, j)]["delta"])] if variant == "delta" else \
                         [np.asarray(feats[(l, j)]["h"]), np.asarray(feats[(l, j)]["delta"])]
                aucs[l] = fit_and_score(blocks, y, fit_mask, args.pca_dim, args.C)
                del blocks
                gc.collect()
            best = max(aucs, key=lambda l: (aucs[l] if aucs[l] == aucs[l] else -1))
            name = {0: "Q->E1", 1: "E1->E2", 2: "E2->E3", 3: "E3->E4"}[j]
            print(f"{j:>3} {name:<10} {len(y):>8} {y.mean():>8.3f} "
                  + "".join(f"{aucs[l]:>11.4f}" for l in layers)
                  + f"{best:>9}{PRODUCTION_LAYERS.get(j, '-'):>8}"
                  + ("" if best == PRODUCTION_LAYERS.get(j) else "   <-- 与生产选择不同"))
            per_j[j] = {"n_rows": int(len(y)), "pos_rate": float(y.mean()), "auc_by_layer": aucs,
                        "best_layer": int(best), "production_layer": PRODUCTION_LAYERS.get(j),
                        "agrees_with_production": bool(best == PRODUCTION_LAYERS.get(j)),
                        "gap_to_production": float(aucs[best] - aucs[PRODUCTION_LAYERS[j]]) if j in PRODUCTION_LAYERS else None}
        report["per_feature"][variant] = per_j
        agree = sum(1 for v in per_j.values() if v["agrees_with_production"])
        print(f"  → {agree}/{len(per_j)} 个位置与生产选择一致\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
