#!/usr/bin/env python3
"""
Fit pooled-by-semantic-transition LR gates on hidden-state delta vectors.

Pooled protocol: one model per semantic j (Q->E1, E1->E2, ...), pooling K=2,3,4.
Excludes final transition j=K.

Train on hidden_states/train/, evaluate on dev (default) or train.

Usage:
  python fit_lr_gate_pooled.py
  python fit_lr_gate_pooled.py --hs-root ../hidden_states --target-fpr 0.15
  python apply_lr_gate_pooled.py --split dev
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
PROJECT_ROOT = HERE.parent
HS_ROOT = PROJECT_ROOT / "hidden_states"
RESULTS_DIR = HERE / "results_pooled"
ARTIFACTS_DIR = HERE / "artifacts_pooled"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fit_lr_gate import collect_split, transition_label
from hop_labels import err_hop_bucket_for_row


def pooled_transition_label(j: int) -> str:
    if j == 0:
        return "Q->E1"
    return f"E{j}->E{j + 1}"


def collect_split_pooled(
    hs_root: Path,
    split: str,
    *,
    max_j: int = 3,
    layer: int | None = None,
    layer_by_j: dict[int, int] | None = None,
) -> dict[int, dict]:
    """
    Pool hidden-state deltas by semantic transition j, across every K > j.

    Two mutually exclusive modes:
    - `layer` (default None = legacy single-layer directories): one fixed
      layer for every j — reads the split once.
    - `layer_by_j`: a different layer per transition (e.g. {0: 15, 1: 15,
      2: 23, 3: 23}, because different layers turned out to carry the
      strongest signal for different transitions — see 0705update.md).
      Each distinct layer is read from disk once and cached even if it's
      reused across multiple j's, then only that layer's rows for its
      assigned j are kept.
    """
    if layer_by_j:
        pooled: dict[int, dict] = defaultdict(lambda: {"X": [], "y": [], "meta": []})
        cache: dict[int, dict[tuple[int, int], dict]] = {}
        for j, lyr in sorted(layer_by_j.items()):
            if lyr not in cache:
                cache[lyr] = collect_split(hs_root, split, layer=lyr)
            per_kj = cache[lyr]
            for (K, jj), group in sorted(per_kj.items()):
                if jj != j or jj >= K:
                    continue
                for delta, label, meta in zip(group["X"], group["y"], group["meta"]):
                    pooled[j]["X"].append(delta)
                    pooled[j]["y"].append(label)
                    pooled[j]["meta"].append(dict(meta))
        return pooled

    per_kj = collect_split(hs_root, split, layer=layer)
    pooled: dict[int, dict] = defaultdict(lambda: {"X": [], "y": [], "meta": []})
    for (K, j), group in sorted(per_kj.items()):
        if j >= K or j > max_j:
            continue
        for delta, label, meta in zip(group["X"], group["y"], group["meta"]):
            pooled[j]["X"].append(delta)
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


def aggregate_metrics(score_rows: list[dict], *, auc: float, target_fpr: float) -> dict:
    out: dict = {
        "auc": round(float(auc), 4),
        "target_fpr": target_fpr,
        "threshold_source": "train_calibrated_per_pooled_j",
        "pooling": "semantic_j_across_K_excluding_final",
        "overall": compute_metrics(score_rows),
    }
    by_j: dict[int, list] = collections.defaultdict(list)
    by_K: dict[int, list] = collections.defaultdict(list)
    by_hop: dict[str, list] = collections.defaultdict(list)
    by_K_hop: dict[tuple[int, str], list] = collections.defaultdict(list)
    for r in score_rows:
        by_j[int(r["j"])].append(r)
        by_K[int(r["K"])].append(r)
        bucket = err_hop_bucket_for_row(
            r.get("wrong_hops") or [], int(r["j"]), trace_type=str(r["trace_type"]),
        )
        by_hop[bucket].append(r)
        by_K_hop[(int(r["K"]), bucket)].append(r)
    out["by_j"] = {str(j): compute_metrics(v) for j, v in sorted(by_j.items())}
    out["by_K"] = {str(K): compute_metrics(v) for K, v in sorted(by_K.items())}
    out["by_err_hop"] = {k: compute_metrics(v) for k, v in sorted(by_hop.items())}
    out["by_K_err_hop"] = {
        f"K{K}_{bucket}": compute_metrics(v)
        for (K, bucket), v in sorted(by_K_hop.items())
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit pooled-by-semantic-j LR delta gate.")
    ap.add_argument("--hs-root", type=Path, default=HS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--split-eval", choices=("dev", "train"), default="dev")
    ap.add_argument("--target-fpr", type=float, default=0.15)
    ap.add_argument("--max-j", type=int, default=3)
    ap.add_argument("--no-save-artifacts", action="store_true")
    ap.add_argument("--layer", type=int, default=None,
                    help="Which transformer layer to read. Leave unset for legacy single-layer "
                         "directories (hidden_states/{train,dev}/). Required when --hs-root points "
                         "at a multi-layer pilot directory (hidden_states/pilot_multilayer/) — a "
                         "layer only needs to be extracted once there, not duplicated into a "
                         "separate single-layer directory just to retrain the pooled gate on it.")
    ap.add_argument("--train-split-name", default="train",
                    help="Subdirectory name under --hs-root holding the train split. Production "
                         "layout uses 'train'; the multi-layer pilot extractor uses 'train_pilot'.")
    ap.add_argument("--per-j-layers", default=None,
                    help="Mix a different layer per transition instead of one fixed --layer for "
                         "all of them, e.g. '0:15,1:15,2:23,3:23'. Each j gets its own independent "
                         "model regardless (pooled protocol already fits j=0..3 separately), so "
                         "nothing requires them to share a layer — see gate/compare_layers_pilot.py "
                         "and 0705update.md for how this was chosen. Takes precedence over --layer "
                         "when given; the layer actually used for each j is recorded in that j's "
                         "saved artifact (and in meta.json) so downstream readers don't need to be "
                         "told separately which layer goes with which hop.")
    args = ap.parse_args()

    if not (args.hs_root / args.train_split_name / "activations").is_dir():
        sys.exit(
            f"No training activations at {args.hs_root}/{args.train_split_name}/activations/\n"
            "Run hidden_states/extract_hidden_states.py first."
        )

    layer_by_j: dict[int, int] | None = None
    if args.per_j_layers:
        layer_by_j = {}
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
    print("multihop_trace pooled LR training")
    print(f"  hs-root:       {args.hs_root}")
    print(f"  artifacts-dir: {args.artifacts_dir}")
    print(f"  split-eval:    {args.split_eval}")
    print(f"  target-fpr:    {args.target_fpr}")
    if layer_by_j:
        print(f"  per-j layers:  {sorted(layer_by_j.items())}")
    else:
        print(f"  layer:         {args.layer}")
    print("=" * 60)

    print("\nLoading train split ...", flush=True)
    train_data = collect_split_pooled(
        args.hs_root, args.train_split_name, max_j=args.max_j,
        layer=args.layer, layer_by_j=layer_by_j,
    )
    print(f"  pooled j groups: {len(train_data)}")

    print(f"Loading {args.split_eval} split ...", flush=True)
    eval_data = collect_split_pooled(
        args.hs_root, args.split_eval, max_j=args.max_j,
        layer=args.layer, layer_by_j=layer_by_j,
    )
    print(f"  pooled j groups: {len(eval_data)}")

    print(f"\nFitting pooled LR (pca_dim={args.pca_dim}, C={args.C}) ...")
    models: dict[int, dict] = {}
    for j, td in tqdm(sorted(train_data.items()), desc="fit", unit="group"):
        X_tr = np.stack(td["X"], axis=0)
        y_tr = np.array(td["y"], dtype=int)
        n, d = X_tr.shape
        n_pos = int(y_tr.sum())
        K_used = sorted({int(m["K"]) for m in td["meta"]})
        if n_pos < 5 or (n - n_pos) < 5:
            continue

        nc = min(args.pca_dim, n - 1, d)
        pca = PCA(n_components=nc, random_state=0)
        Z_tr = pca.fit_transform(X_tr.astype(np.float64))
        lr = LogisticRegression(
            C=args.C, class_weight="balanced", max_iter=1000, random_state=0, solver="lbfgs",
        )
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
            "pca": pca,
            "lr": lr,
            "nc": nc,
            "threshold": tau,
            "train_auc": round(float(roc_auc_score(y_tr, train_proba)), 4),
            "train_roc": {
                "fpr": fpr_arr.tolist(),
                "tpr": tpr_arr.tolist(),
                "thresholds": thr_arr.tolist(),
            },
            "K_used": K_used,
        }
        print(
            f"  j={j} {pooled_transition_label(j):>10} K={K_used} "
            f"n={n} pos={100*n_pos/n:.1f}% tau={tau:.3f}"
        )

    hidden_dim = None
    for td in train_data.values():
        if td["X"]:
            hidden_dim = int(np.asarray(td["X"][0]).shape[0])
            break

    if not args.no_save_artifacts:
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        meta_rows = []
        for j, m in sorted(models.items()):
            # The layer actually used for THIS j — from --per-j-layers if given,
            # else the one global --layer. Recorded on the artifact itself (not
            # just in meta.json) so a single j{}.joblib is self-describing even
            # if copied out on its own: downstream readers (e.g. the retrieval
            # gate scorer) don't need a separate lookup table telling them which
            # layer this model expects its input Δ to come from.
            j_layer = layer_by_j[j] if layer_by_j else args.layer
            fname = f"j{j}.joblib"
            joblib.dump({
                "pooling": "semantic_j_across_K_excluding_final",
                "j": j,
                "transition": pooled_transition_label(j),
                "K_used": m["K_used"],
                "pca": m["pca"],
                "lr": m["lr"],
                "threshold": m["threshold"],
                "pca_dim": m["nc"],
                "pca_fit_on": "all_train_pooled_by_j",
                "target_fpr": args.target_fpr,
                "C": args.C,
                "train_split": args.train_split_name,
                "layer": j_layer,
                "hs_root": str(args.hs_root),
                "hidden_dim": hidden_dim,
                "created_time": datetime.datetime.now().isoformat(timespec="seconds"),
                "train_auc": m["train_auc"],
                "train_roc": m["train_roc"],
            }, args.artifacts_dir / fname)
            meta_rows.append({
                "j": j,
                "transition": pooled_transition_label(j),
                "layer": j_layer,
                "K_used": m["K_used"],
                "pca_dim": m["nc"],
                "threshold": m["threshold"],
                "train_auc": m["train_auc"],
                "file": fname,
            })
        (args.artifacts_dir / "meta.json").write_text(json.dumps({
            "pooling": "semantic_j_across_K_excluding_final",
            "n_models": len(models),
            "pca_dim": args.pca_dim,
            "C": args.C,
            "target_fpr": args.target_fpr,
            "layer": args.layer,
            "per_j_layers": layer_by_j,
            "hidden_dim": hidden_dim,
            "hs_root": str(args.hs_root),
            "created_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "models": meta_rows,
        }, indent=2), encoding="utf-8")
        print(f"\nArtifacts -> {args.artifacts_dir} ({len(models)} models)")

    print(f"\nEvaluating on {args.split_eval} ...")
    score_rows: list[dict] = []
    for j, ed in sorted(eval_data.items()):
        if j not in models:
            continue
        m = models[j]
        proba = m["lr"].predict_proba(
            m["pca"].transform(np.stack(ed["X"]).astype(np.float64))
        )[:, 1]
        tau = float(m["threshold"])
        for meta, p, label in zip(ed["meta"], proba, ed["y"]):
            K = int(meta["K"])
            score_rows.append({
                "id": meta["id"],
                "split": args.split_eval,
                "K": K,
                "j": j,
                "transition": transition_label(j, K),
                "pooled_transition": pooled_transition_label(j),
                "pooling": "semantic_j_across_K_excluding_final",
                "trace_type": meta["trace_type"],
                "wrong_hops": meta["wrong_hops"],
                "score": round(float(p), 5),
                "should_intervene": bool(label),
                "threshold": round(tau, 5),
                "triggered": bool(p > tau),
                "distance": round(float(p), 5),
            })

    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(
        [r["should_intervene"] for r in score_rows],
        [r["score"] for r in score_rows],
    ))
    print(f"ROC-AUC (pooled LR, {args.split_eval}): {auc:.4f}")

    out_scores = args.out_dir / f"{args.split_eval}_lr_pooled_scores.jsonl"
    with out_scores.open("w", encoding="utf-8") as f:
        for r in score_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = aggregate_metrics(score_rows, auc=auc, target_fpr=args.target_fpr)
    out_metrics = args.out_dir / f"{args.split_eval}_lr_pooled_metrics.json"
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Scores -> {out_scores}")
    print(f"Metrics -> {out_metrics}")

    ov = metrics["overall"]
    print(f"\n== Overall ({args.split_eval}) ==")
    print(f"  AUC={auc:.4f}  TPR={ov['TPR']:.3f}  FPR={ov['FPR']:.3f}  F1={ov['F1']:.3f}")


if __name__ == "__main__":
    main()
