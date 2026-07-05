#!/usr/bin/env python3
"""
Fit per-(K, transition) Logistic Regression gate on delta vectors.

Train: hidden_states/train/  →  PCA + LR + train-calibrated threshold
Eval:  hidden_states/dev/    →  scores + metrics

Usage:
  python fit_lr_gate.py
  python fit_lr_gate.py --split-eval dev --target-fpr 0.15
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
HS_ROOT = PROJECT_ROOT / "hidden_states"
RESULTS_DIR = HERE / "results"
ARTIFACTS_DIR = HERE / "artifacts"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hop_labels import err_hop_bucket_for_row, parse_wrong_hops, should_intervene_at_j


def transition_label(j: int, K: int) -> str:
    if j == 0:
        return "Q->E1"
    if j == K:
        return f"E{K}->Final"
    return f"E{j}->E{j + 1}"


def load_manifest_meta(manifest_path: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if not manifest_path.is_file():
        return meta
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = str(row.get("id", "")).strip()
            if rid:
                meta[rid] = row
    return meta


def read_K_from_npz(z, hidden) -> int:
    import numpy as np

    if "K" in z:
        return int(np.asarray(z["K"]).reshape(-1)[0])
    # multihop_trace: K+1 prefix hiddens for K-hop trace
    return int(hidden.shape[0]) - 1


def read_hidden_from_npz(z, *, layer: int | None):
    """
    Read the (T, hidden_dim) last-token hidden array out of an npz, handling
    both storage formats that exist in this repo:

    - legacy single-layer (hidden_states/{train,dev}/, from extract_hidden_states.py):
      `hidden` is already (T, hidden_dim) for whatever single layer was
      extracted (production default: layer 31). `layer` must be left None
      here — there's nothing to select between.
    - multi-layer pilot (hidden_states/pilot_multilayer/, from
      extract_hidden_states_multilayer_pilot.py): `hidden` is stacked as
      (n_layers, T, hidden_dim) with a parallel `layers` array recording
      which transformer layer each slice is. `layer` must be given so this
      knows which slice to pull out.

    Letting the same collect_split() read either format means a layer only
    needs to be extracted once (in the multi-layer format) rather than kept
    twice — once in a single-layer directory and again inside the pilot's
    multi-layer one just to compare it.
    """
    import numpy as np

    if "layers" in z.files:
        if layer is None:
            raise ValueError(
                "npz is in multi-layer pilot format (has a 'layers' field) but no `layer` was "
                "given to collect_split() — specify which layer to read."
            )
        layers_arr = [int(x) for x in np.asarray(z["layers"]).reshape(-1)]
        if layer not in layers_arr:
            return None
        layer_pos = layers_arr.index(layer)
        return z["hidden"][layer_pos].astype(np.float32)

    if layer is not None:
        raise ValueError(
            f"npz is in legacy single-layer format (no 'layers' field) but layer={layer} was "
            "given — legacy directories only ever hold one layer; drop `layer` (leave it None)."
        )
    return z["hidden"].astype(np.float32)


def collect_split(
    hs_root: Path,
    split: str,
    subdirs: tuple[str, ...] = ("pos", "neg"),
    layer: int | None = None,
) -> dict[tuple[int, int], dict]:
    """
    Returns {(K, j): {"X": [delta...], "y": [label...], "meta": [...]}}.

    `layer`: which transformer layer to read. None (default) preserves the
    original behavior for legacy single-layer directories
    (hidden_states/{train,dev}/). Pass an int to read that layer out of a
    multi-layer pilot directory (hidden_states/pilot_multilayer/) instead.
    """
    import numpy as np

    manifest = load_manifest_meta(hs_root / split / "manifest.jsonl")
    data: dict[tuple[int, int], dict] = defaultdict(lambda: {"X": [], "y": [], "meta": []})

    for sd in subdirs:
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
                delta = (hidden[j + 1] - hidden[j]).astype(np.float32)
                label = should_intervene_at_j(wrong_hops, j, trace_type=trace_type)
                data[(K, j)]["X"].append(delta)
                data[(K, j)]["y"].append(label)
                data[(K, j)]["meta"].append({
                    "id": eid,
                    "trace_type": trace_type,
                    "wrong_hops": wrong_hops,
                    "K": K,
                    "j": j,
                })
    return data


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
    ap = argparse.ArgumentParser(description="Fit per-(K,j) LR delta gate.")
    ap.add_argument("--hs-root", type=Path, default=HS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--split-eval", choices=("dev", "train"), default="dev")
    ap.add_argument("--target-fpr", type=float, default=0.15)
    ap.add_argument("--no-save-artifacts", action="store_true")
    ap.add_argument("--layer", type=int, default=None,
                    help="Which transformer layer to read. Leave unset for legacy single-layer "
                         "directories (hidden_states/{train,dev}/). Required when --hs-root points "
                         "at a multi-layer pilot directory (hidden_states/pilot_multilayer/).")
    ap.add_argument("--train-split-name", default="train",
                    help="Subdirectory name under --hs-root holding the train split. Production "
                         "layout uses 'train'; the multi-layer pilot extractor uses 'train_pilot'.")
    args = ap.parse_args()

    try:
        import joblib
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, roc_curve
        from tqdm import tqdm
    except ImportError as exc:
        sys.exit(f"Install: pip install numpy scikit-learn joblib tqdm ({exc})")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train split ...", flush=True)
    train_data = collect_split(args.hs_root, args.train_split_name, layer=args.layer)
    print(f"  (K, j) groups: {len(train_data)}")

    print(f"Loading {args.split_eval} split ...", flush=True)
    eval_data = collect_split(args.hs_root, args.split_eval, layer=args.layer)
    print(f"  (K, j) groups: {len(eval_data)}")

    print(f"\nFitting LR (pca_dim={args.pca_dim}, C={args.C}) ...")
    models: dict[tuple[int, int], dict] = {}

    for (K, j), td in tqdm(sorted(train_data.items()), desc="fit", unit="group"):
        X_tr = np.stack(td["X"], axis=0)
        y_tr = np.array(td["y"], dtype=int)
        n, d = X_tr.shape
        n_pos = int(y_tr.sum())
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
        models[(K, j)] = {
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
        }

    print(f"Models ready: {len(models)} / {len(train_data)}")

    hidden_dim = None
    for td in train_data.values():
        if td["X"]:
            hidden_dim = int(np.asarray(td["X"][0]).shape[0])
            break

    if not args.no_save_artifacts:
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        meta_rows = []
        for (K, j), m in sorted(models.items()):
            fname = f"K{K}_j{j}.joblib"
            joblib.dump({
                "pca": m["pca"],
                "lr": m["lr"],
                "threshold": m["threshold"],
                "K": K,
                "j": j,
                "transition": transition_label(j, K),
                "pca_dim": m["nc"],
                "pca_fit_on": "all_train",
                "target_fpr": args.target_fpr,
                "C": args.C,
                "train_split": "train",
                "hs_root": str(args.hs_root),
                "hidden_dim": hidden_dim,
                "created_time": datetime.datetime.now().isoformat(timespec="seconds"),
                "train_auc": m["train_auc"],
                "train_roc": m["train_roc"],
            }, args.artifacts_dir / fname)
            meta_rows.append({
                "K": K, "j": j, "transition": transition_label(j, K),
                "pca_dim": m["nc"], "threshold": m["threshold"],
                "train_auc": m["train_auc"], "file": fname,
            })
        (args.artifacts_dir / "meta.json").write_text(json.dumps({
            "n_models": len(models),
            "pca_dim": args.pca_dim,
            "C": args.C,
            "target_fpr": args.target_fpr,
            "hidden_dim": hidden_dim,
            "hs_root": str(args.hs_root),
            "created_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "models": meta_rows,
        }, indent=2), encoding="utf-8")

    print(f"\nEvaluating on {args.split_eval} ...")
    score_rows: list[dict] = []
    for (K, j), ed in sorted(eval_data.items()):
        if (K, j) not in models:
            continue
        m = models[(K, j)]
        proba = m["lr"].predict_proba(m["pca"].transform(np.stack(ed["X"]).astype(np.float64)))[:, 1]
        tau = m["threshold"]
        for meta, p, label in zip(ed["meta"], proba, ed["y"]):
            score_rows.append({
                "id": meta["id"],
                "split": args.split_eval,
                "K": K,
                "j": j,
                "transition": transition_label(j, K),
                "trace_type": meta["trace_type"],
                "wrong_hops": meta["wrong_hops"],
                "score": round(float(p), 5),
                "should_intervene": bool(label),
                "threshold": round(tau, 5),
                "triggered": bool(p > tau),
                "distance": round(float(p), 5),
            })

    auc = float(roc_auc_score([r["should_intervene"] for r in score_rows],
                              [r["score"] for r in score_rows]))
    print(f"ROC-AUC (LR, {args.split_eval}): {auc:.4f}")

    out_scores = args.out_dir / f"{args.split_eval}_lr_scores.jsonl"
    with out_scores.open("w", encoding="utf-8") as f:
        for r in score_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    all_metrics: dict = {
        "auc": round(auc, 4),
        "target_fpr": args.target_fpr,
        "threshold_source": "train_calibrated",
        "overall": compute_metrics(score_rows),
    }
    by_K: dict[int, list] = defaultdict(list)
    by_hop: dict[str, list] = defaultdict(list)
    by_K_hop: dict[tuple[int, str], list] = defaultdict(list)
    for r in score_rows:
        by_K[r["K"]].append(r)
        bucket = err_hop_bucket_for_row(r.get("wrong_hops") or [], r["j"], trace_type=r["trace_type"])
        by_hop[bucket].append(r)
        by_K_hop[(r["K"], bucket)].append(r)
    all_metrics["by_K"] = {str(K): compute_metrics(v) for K, v in sorted(by_K.items())}
    all_metrics["by_err_hop"] = {k: compute_metrics(v) for k, v in sorted(by_hop.items())}
    all_metrics["by_K_err_hop"] = {
        f"K{K}_{bucket}": compute_metrics(v) for (K, bucket), v in sorted(by_K_hop.items())
    }
    out_metrics = args.out_dir / f"{args.split_eval}_lr_metrics.json"
    out_metrics.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"Scores -> {out_scores}")
    print(f"Metrics -> {out_metrics}")


if __name__ == "__main__":
    main()
