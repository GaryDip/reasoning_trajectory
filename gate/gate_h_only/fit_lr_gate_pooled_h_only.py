#!/usr/bin/env python3
"""
Gate h-only: uses ONLY the absolute hidden state after each hop (h_after = hidden[j+1],
"is the accumulated state reasonable so far") as the classifier's input feature -- no
Delta_j term at all. This is the third point of the 3-way comparison the other two gates
already cover: gate v2 (gate/fit_lr_gate_pooled.py) is Delta_j only, gate v3
(gate/gate_v3/fit_lr_gate_pooled_v3.py) is [PCA(h_after), PCA(Delta_j)] concatenated. This
script is deliberately the same pooled-per-j PCA+LR architecture as those two, differing
ONLY in which feature(s) go in, so the three are a controlled, apples-to-apples comparison
-- not e.g. error_propagation_probe's h_j-probe, which uses a different pooling/training
protocol and is not directly comparable.

    p_j = sigmoid(w_h^T . PCA(h_after) + b)

No new Llama extraction needed -- hidden_states/pilot_multilayer/{train_pilot,dev}/ already
stores the full per-trace hidden-state sequence h_after is read from (same source gate v2's
Delta_j and gate v3's h_after both already use). Reuses (imports, does not copy)
gate/fit_lr_gate.py's manifest/label helpers and gate/hop_labels.py -- same as gate v2 and
gate v3 do; none of those three scripts modify each other.

Same per-j layer choice gate v2 already validated: j=0,1 -> layer 15, j=2,3 -> layer 23.

Usage:
  python fit_lr_gate_pooled_h_only.py
  python fit_lr_gate_pooled_h_only.py --pca-dim-h 64
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GATE_DIR = HERE.parent
PROJECT_ROOT = GATE_DIR.parent
HS_ROOT = PROJECT_ROOT / "hidden_states" / "pilot_multilayer"
RESULTS_DIR = HERE / "results_pooled_h_only"
ARTIFACTS_DIR = HERE / "artifacts_pooled_h_only"

sys.path.insert(0, str(GATE_DIR))

from fit_lr_gate import load_manifest_meta, read_hidden_from_npz, read_K_from_npz  # noqa: E402
from hop_labels import err_hop_bucket_for_row, parse_wrong_hops, should_intervene_at_j  # noqa: E402

DEFAULT_PER_J_LAYERS = {0: 15, 1: 15, 2: 23, 3: 23}


def pooled_transition_label(j: int) -> str:
    if j == 0:
        return "Q->E1"
    return f"E{j}->E{j + 1}"


def collect_split(hs_root: Path, split: str, *, layer: int) -> dict[tuple[int, int], dict]:
    """Like fit_lr_gate.py::collect_split, but keeps h_after (=hidden[j+1], the absolute
    state right after transition j) instead of Delta_j."""
    manifest = load_manifest_meta(hs_root / split / "manifest.jsonl")
    data: dict[tuple[int, int], dict] = defaultdict(lambda: {"X_h": [], "y": [], "meta": []})

    for sd in ("pos", "neg"):
        npz_dir = hs_root / split / "activations" / sd
        if not npz_dir.is_dir():
            continue
        trace_type = "correct" if sd == "pos" else "error"
        for f in sorted(npz_dir.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            hidden = read_hidden_from_npz(z, layer=layer)
            if hidden is None:
                continue
            eid = str(np.asarray(z["example_id"]).reshape(-1)[0]).strip()
            if not eid:
                continue
            T = hidden.shape[0]
            K = read_K_from_npz(z, hidden)
            if K < 1:
                continue

            mrow = manifest.get(eid)
            wrong_hops = parse_wrong_hops(mrow, z)

            for j in range(T - 1):
                if j >= K:
                    continue
                h_after = hidden[j + 1].astype(np.float32)
                label = should_intervene_at_j(wrong_hops, j, trace_type=trace_type)
                data[(K, j)]["X_h"].append(h_after)
                data[(K, j)]["y"].append(label)
                data[(K, j)]["meta"].append({
                    "id": eid, "trace_type": trace_type, "wrong_hops": wrong_hops, "K": K, "j": j,
                })
    return data


def collect_split_pooled(hs_root: Path, split: str, *, max_j: int, layer_by_j: dict[int, int]) -> dict[int, dict]:
    pooled: dict[int, dict] = defaultdict(lambda: {"X_h": [], "y": [], "meta": []})
    cache: dict[int, dict[tuple[int, int], dict]] = {}
    for j, lyr in sorted(layer_by_j.items()):
        if j > max_j:
            continue
        if lyr not in cache:
            cache[lyr] = collect_split(hs_root, split, layer=lyr)
        per_kj = cache[lyr]
        for (K, jj), group in sorted(per_kj.items()):
            if jj != j or jj >= K:
                continue
            for h_after, label, meta in zip(group["X_h"], group["y"], group["meta"]):
                pooled[j]["X_h"].append(h_after)
                pooled[j]["y"].append(label)
                pooled[j]["meta"].append(dict(meta))
    return pooled


def compute_metrics(rows: list[dict]) -> dict:
    tp = sum(1 for r in rows if r["triggered"] and r["should_intervene"])
    fp = sum(1 for r in rows if r["triggered"] and not r["should_intervene"])
    fn = sum(1 for r in rows if not r["triggered"] and r["should_intervene"])
    tn = sum(1 for r in rows if not r["triggered"] and not r["should_intervene"])
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0.0
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "TPR": round(tpr, 4), "FPR": round(fpr, 4),
        "Precision": round(prec, 4), "F1": round(f1, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hs-root", type=Path, default=HS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    ap.add_argument("--pca-dim-h", type=int, default=64)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--split-eval", choices=("dev", "train"), default="dev")
    ap.add_argument("--target-fpr", type=float, default=0.15)
    ap.add_argument("--max-j", type=int, default=3)
    ap.add_argument("--train-split-name", default="train_pilot")
    ap.add_argument("--per-j-layers", default="0:15,1:15,2:23,3:23",
                     help="Same per-j layer choice gate v2/v3 already validated — see module docstring.")
    ap.add_argument("--no-save-artifacts", action="store_true")
    args = ap.parse_args()

    if not (args.hs_root / args.train_split_name / "activations").is_dir():
        sys.exit(f"No training activations at {args.hs_root}/{args.train_split_name}/activations/")

    layer_by_j: dict[int, int] = {}
    for part in args.per_j_layers.split(","):
        part = part.strip()
        if not part:
            continue
        j_str, layer_str = part.split(":")
        layer_by_j[int(j_str)] = int(layer_str)

    try:
        import joblib
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, roc_curve
        from tqdm import tqdm
    except ImportError as exc:
        sys.exit(f"Install: pip install numpy scikit-learn joblib tqdm ({exc})")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("gate h-only: PCA(h_after) only, one LR per pooled j")
    print(f"  hs-root:       {args.hs_root}")
    print(f"  artifacts-dir: {args.artifacts_dir}")
    print(f"  split-eval:    {args.split_eval}")
    print(f"  per-j layers:  {sorted(layer_by_j.items())}")
    print(f"  pca_dim_h:     {args.pca_dim_h}")
    print("=" * 60)

    print("\nLoading train split ...", flush=True)
    train_data = collect_split_pooled(args.hs_root, args.train_split_name, max_j=args.max_j, layer_by_j=layer_by_j)
    print(f"  pooled j groups: {len(train_data)}")

    print(f"Loading {args.split_eval} split ...", flush=True)
    eval_data = collect_split_pooled(args.hs_root, args.split_eval, max_j=args.max_j, layer_by_j=layer_by_j)
    print(f"  pooled j groups: {len(eval_data)}")

    print(f"\nFitting pooled LR (PCA on h_after only) ...")
    models: dict[int, dict] = {}
    for j, td in tqdm(sorted(train_data.items()), desc="fit", unit="group"):
        X_h_tr = np.stack(td["X_h"], axis=0).astype(np.float64)
        y_tr = np.array(td["y"], dtype=int)
        n = len(y_tr)
        n_pos = int(y_tr.sum())
        K_used = sorted({int(m["K"]) for m in td["meta"]})
        if n_pos < 5 or (n - n_pos) < 5:
            continue

        nc_h = min(args.pca_dim_h, n - 1, X_h_tr.shape[1])
        pca_h = PCA(n_components=nc_h, random_state=0)
        Z_tr = pca_h.fit_transform(X_h_tr)

        lr = LogisticRegression(C=args.C, class_weight="balanced", max_iter=1000, random_state=0, solver="lbfgs")
        lr.fit(Z_tr, y_tr)

        train_proba = lr.predict_proba(Z_tr)[:, 1]
        neg_scores = train_proba[y_tr == 0]
        tau = float(np.max(train_proba))
        for c in np.sort(np.unique(train_proba)):
            if float(np.mean(neg_scores > c)) <= args.target_fpr:
                tau = float(c)
                break

        fpr_arr, tpr_arr, thr_arr = roc_curve(y_tr, train_proba)
        models[j] = {
            "pca_h": pca_h, "lr": lr,
            "nc_h": nc_h,
            "threshold": tau,
            "train_auc": round(float(roc_auc_score(y_tr, train_proba)), 4),
            "layer": layer_by_j[j],
            "K_used": K_used,
        }
        print(f"  j={j} {pooled_transition_label(j):>10} K={K_used} n={n} pos={100*n_pos/n:.1f}% "
              f"tau={tau:.3f} train_auc={models[j]['train_auc']}")

    if not args.no_save_artifacts:
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        meta_rows = []
        for j, m in models.items():
            fname = f"j{j}.joblib"
            joblib.dump(
                {"pca_h": m["pca_h"], "lr": m["lr"],
                 "threshold": m["threshold"], "layer": m["layer"]},
                args.artifacts_dir / fname,
            )
            meta_rows.append({
                "j": j, "transition": pooled_transition_label(j), "layer": m["layer"],
                "K_used": m["K_used"], "nc_h": m["nc_h"],
                "threshold": m["threshold"], "train_auc": m["train_auc"], "file": fname,
            })
        meta = {
            "pooling": "semantic_j_across_K_excluding_final",
            "feature": "PCA(h_after) only",
            "n_models": len(models),
            "pca_dim_h": args.pca_dim_h,
            "C": args.C, "target_fpr": args.target_fpr,
            "per_j_layers": layer_by_j,
            "hidden_dim": 4096,
            "hs_root": str(args.hs_root),
            "created_time": datetime.datetime.now().isoformat(),
            "models": meta_rows,
        }
        (args.artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"\nSaved artifacts -> {args.artifacts_dir}")

    print(f"\nScoring {args.split_eval} ...")
    score_rows = []
    for j, ed in sorted(eval_data.items()):
        if j not in models:
            continue
        m = models[j]
        X_h_ev = np.stack(ed["X_h"], axis=0).astype(np.float64)
        Z_ev = m["pca_h"].transform(X_h_ev)
        proba = m["lr"].predict_proba(Z_ev)[:, 1]
        for p, y, meta in zip(proba, ed["y"], ed["meta"]):
            score_rows.append({
                **meta, "score": float(p), "triggered": bool(p > m["threshold"]), "should_intervene": bool(y),
            })

    y_all = np.array([1 if r["should_intervene"] else 0 for r in score_rows])
    p_all = np.array([r["score"] for r in score_rows])
    auc = roc_auc_score(y_all, p_all) if len(set(y_all.tolist())) > 1 else float("nan")

    by_j: dict[int, list] = collections.defaultdict(list)
    by_K: dict[int, list] = collections.defaultdict(list)
    by_hop: dict[str, list] = collections.defaultdict(list)
    for r in score_rows:
        by_j[int(r["j"])].append(r)
        by_K[int(r["K"])].append(r)
        bucket = err_hop_bucket_for_row(r.get("wrong_hops") or [], int(r["j"]), trace_type=str(r["trace_type"]))
        by_hop[bucket].append(r)

    results = {
        "auc": round(float(auc), 4),
        "target_fpr": args.target_fpr,
        "overall": compute_metrics(score_rows),
        "by_j": {str(j): compute_metrics(v) for j, v in sorted(by_j.items())},
        "by_K": {str(K): compute_metrics(v) for K, v in sorted(by_K.items())},
        "by_err_hop": {k: compute_metrics(v) for k, v in sorted(by_hop.items())},
    }
    out_path = args.out_dir / f"{args.split_eval}_lr_pooled_h_only_metrics.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\noverall: {results['overall']}  auc={results['auc']}")
    print("by_j:")
    for j, m in results["by_j"].items():
        print(f"  j={j}: {m}")
    print(f"\nSaved -> {out_path}")
    print("\nCompare against gate v2 (Delta-only): gate/results_pooled_v2/dev_lr_pooled_metrics.json")
    print("Compare against gate v3 (h+Delta fused): gate/gate_v3/results_pooled_v3/dev_lr_pooled_v3_metrics.json")


if __name__ == "__main__":
    main()
