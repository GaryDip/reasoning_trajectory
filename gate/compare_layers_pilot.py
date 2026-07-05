#!/usr/bin/env python3
"""
Pilot: compare per-layer pooled LR gates fit on the multi-layer pilot
extraction (hidden_states/extract_hidden_states_multilayer_pilot.py).

For each candidate layer, fits one PCA(64)+LogisticRegression per pooled
semantic transition j (same protocol as gate/fit_lr_gate_pooled.py: pooled
across K, excluding the final transition), trains on hidden_states/pilot_multilayer/train_pilot/,
evaluates on hidden_states/pilot_multilayer/dev/, and prints/saves a
layer x transition AUC table so you can see whether the production choice
(last layer, 31) is actually the strongest layer, or whether an earlier/middle
layer already carries as much (or more) of the "does this evidence fit the
chain" signal.

This is a probe-capacity-matched comparison: every layer uses the exact same
PCA dim + linear LR, only the input features differ. Not meant to replace
gate/fit_lr_gate_pooled.py — this writes to gate/pilot_layer_compare/, not
gate/artifacts_pooled/.

`--concat-layers` additionally fits one PCA+LR per transition on the
*concatenation* of several layers' delta vectors (e.g. layer 15's Δ and layer
23's Δ stacked into one vector before PCA) — still the same simple linear
model, just richer input, to see whether combining layers beats any single
one alone.

Usage:
  python compare_layers_pilot.py
  python compare_layers_pilot.py --layers 15,31 --pca-dim 64
  python compare_layers_pilot.py --concat-layers 15,23
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PILOT_HS_ROOT = PROJECT_ROOT / "hidden_states" / "pilot_multilayer"
OUT_DIR = HERE / "pilot_layer_compare"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fit_lr_gate import load_manifest_meta
from fit_lr_gate_pooled import compute_metrics, pooled_transition_label
from hop_labels import err_hop_bucket_for_row, parse_wrong_hops, should_intervene_at_j


def collect_split_multilayer(hs_root: Path, split: str, layer: int) -> dict[tuple[int, int], dict]:
    """Same shape as fit_lr_gate.collect_split, but reads one layer out of the
    pilot's stacked (n_layers, n_take, hidden_dim) `hidden` array."""
    import numpy as np

    manifest = load_manifest_meta(hs_root / split / "manifest.jsonl")
    data: dict[tuple[int, int], dict] = defaultdict(lambda: {"X": [], "y": [], "meta": []})

    for sd in ("pos", "neg"):
        npz_dir = hs_root / split / "activations" / sd
        if not npz_dir.is_dir():
            continue
        trace_type = "correct" if sd == "pos" else "error"
        for f in sorted(npz_dir.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            layers_arr = [int(x) for x in np.asarray(z["layers"]).reshape(-1)]
            if layer not in layers_arr:
                continue
            layer_pos = layers_arr.index(layer)
            hidden = z["hidden"][layer_pos].astype(np.float32)  # (T, hidden_dim)
            eid = str(np.asarray(z["example_id"]).reshape(-1)[0]).strip()
            if not eid:
                continue
            T = hidden.shape[0]
            K = int(np.asarray(z["K"]).reshape(-1)[0])
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
                    "id": eid, "trace_type": trace_type, "wrong_hops": wrong_hops, "K": K, "j": j,
                })
    return data


def collect_split_concat(hs_root: Path, split: str, layers: list[int]) -> dict[tuple[int, int], dict]:
    """Same as collect_split_multilayer, but concatenates each requested
    layer's delta vector into one longer feature vector per (id, j) — reading
    every layer out of the same npz pass so there's no risk of the per-layer
    feature order drifting apart across separate calls."""
    import numpy as np

    manifest = load_manifest_meta(hs_root / split / "manifest.jsonl")
    data: dict[tuple[int, int], dict] = defaultdict(lambda: {"X": [], "y": [], "meta": []})

    for sd in ("pos", "neg"):
        npz_dir = hs_root / split / "activations" / sd
        if not npz_dir.is_dir():
            continue
        trace_type = "correct" if sd == "pos" else "error"
        for f in sorted(npz_dir.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            layers_arr = [int(x) for x in np.asarray(z["layers"]).reshape(-1)]
            if any(layer not in layers_arr for layer in layers):
                continue
            positions = [layers_arr.index(layer) for layer in layers]
            hidden_all = z["hidden"].astype(np.float32)  # (n_layers, T, hidden_dim)
            eid = str(np.asarray(z["example_id"]).reshape(-1)[0]).strip()
            if not eid:
                continue
            T = hidden_all.shape[1]
            K = int(np.asarray(z["K"]).reshape(-1)[0])
            if K < 1:
                continue

            mrow = manifest.get(eid)
            wrong_hops = parse_wrong_hops(mrow, z)

            for j in range(T - 1):
                if j >= K:
                    continue
                delta = np.concatenate(
                    [(hidden_all[p, j + 1] - hidden_all[p, j]).astype(np.float32) for p in positions]
                )
                label = should_intervene_at_j(wrong_hops, j, trace_type=trace_type)
                data[(K, j)]["X"].append(delta)
                data[(K, j)]["y"].append(label)
                data[(K, j)]["meta"].append({
                    "id": eid, "trace_type": trace_type, "wrong_hops": wrong_hops, "K": K, "j": j,
                })
    return data


def pool_by_j(per_kj: dict[tuple[int, int], dict], *, max_j: int) -> dict[int, dict]:
    pooled: dict[int, dict] = defaultdict(lambda: {"X": [], "y": [], "meta": []})
    for (K, j), group in sorted(per_kj.items()):
        if j >= K or j > max_j:
            continue
        for delta, label, meta in zip(group["X"], group["y"], group["meta"]):
            pooled[j]["X"].append(delta)
            pooled[j]["y"].append(label)
            pooled[j]["meta"].append(dict(meta))
    return pooled


def pool_single_j(per_kj: dict[tuple[int, int], dict], *, j: int) -> dict:
    """Same as pool_by_j but restricted to one transition j, from one layer's
    per_kj dict — the building block for a per-transition best-layer mix."""
    pooled: dict = {"X": [], "y": [], "meta": []}
    for (K, jj), group in sorted(per_kj.items()):
        if jj != j or jj >= K:
            continue
        pooled["X"].extend(group["X"])
        pooled["y"].extend(group["y"])
        pooled["meta"].extend(dict(m) for m in group["meta"])
    return pooled


def make_classifier(clf: str, *, C: float, mlp_hidden: int, mlp_alpha: float):
    from sklearn.linear_model import LogisticRegression

    if clf == "lr":
        return LogisticRegression(
            C=C, class_weight="balanced", max_iter=1000, random_state=0, solver="lbfgs",
        )
    if clf == "mlp":
        from sklearn.neural_network import MLPClassifier

        # Deliberately shallow: one hidden layer, one activation — a probe-
        # capacity-matched step up from linear, not a general-purpose deep
        # model. class_weight isn't available for MLPClassifier, so rely on
        # alpha (L2) + early_stopping to keep it from just memorizing the
        # small pooled-j training sets (as few as ~4k rows at j=3).
        return MLPClassifier(
            hidden_layer_sizes=(mlp_hidden,), activation="relu", alpha=mlp_alpha,
            max_iter=2000, random_state=0, early_stopping=True, n_iter_no_change=20,
        )
    raise ValueError(f"unknown classifier {clf!r}")


def augment_with_squares(Z, k: int):
    """Append squared terms of the top-k (highest-variance) PCA components as
    extra columns — a deliberately tiny, hand-picked bit of nonlinearity for
    LogisticRegression to use, instead of swapping in a whole different model
    class (MLP). A squared term lets the linear model express "extreme
    magnitude on this component predicts the label, regardless of sign" (a
    U-shaped/symmetric relationship a plain linear term can't represent) while
    staying a normal, convex, single-optimum LogisticRegression underneath —
    each added coefficient is individually inspectable (does PC_i² actually
    matter, or is its coefficient ~0), and adding only a handful of columns
    (not a full degree-2 expansion of all pca_dim components, which would run
    to thousands of interaction terms) keeps the model from being powerful
    enough that a good score stops being evidence the trace signal itself is
    separable. k=0 is a no-op (returns Z unchanged)."""
    import numpy as np

    if k <= 0:
        return Z
    k = min(k, Z.shape[1])
    return np.hstack([Z, Z[:, :k] ** 2])


def fit_and_eval_from_per_kj(
    train_per_kj: dict[tuple[int, int], dict],
    dev_per_kj: dict[tuple[int, int], dict],
    *,
    label: str,
    pca_dim: int,
    C: float,
    max_j: int,
    clf: str = "lr",
    mlp_hidden: int = 8,
    mlp_alpha: float = 1.0,
    square_top_k: int = 0,
):
    """Shared pool+fit+eval core for both a single layer and a concatenation
    of several layers — the only difference is what per-(K,j) delta vectors
    were collected before calling this. `clf` swaps the final model (still on
    top of the exact same PCA features) between plain LR and a deliberately
    shallow one-hidden-layer MLP, to test whether a *minimal* amount of
    nonlinearity captures anything the linear probe misses, without jumping
    to a fully expressive model that would muddy whether a result comes from
    the trace signal itself or from classifier capacity. `square_top_k` is an
    alternative, more conservative way to add a little nonlinearity while
    staying strictly LogisticRegression — see augment_with_squares()."""
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    train_pooled = pool_by_j(train_per_kj, max_j=max_j)
    dev_pooled = pool_by_j(dev_per_kj, max_j=max_j)

    per_j: dict[int, dict] = {}
    all_dev_rows: list[dict] = []

    for j, td in sorted(train_pooled.items()):
        if j not in dev_pooled:
            continue
        X_tr = np.stack(td["X"], axis=0)
        y_tr = np.array(td["y"], dtype=int)
        n, d = X_tr.shape
        n_pos = int(y_tr.sum())
        if n_pos < 5 or (n - n_pos) < 5:
            per_j[j] = {"skipped": True, "reason": "not enough pos/neg in train pilot sample", "n_train": n}
            continue

        nc = min(pca_dim, n - 1, d)
        pca = PCA(n_components=nc, random_state=0)
        Z_tr = augment_with_squares(pca.fit_transform(X_tr.astype(np.float64)), square_top_k)
        model = make_classifier(clf, C=C, mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha)
        model.fit(Z_tr, y_tr)

        ed = dev_pooled[j]
        X_dev = np.stack(ed["X"], axis=0).astype(np.float64)
        y_dev = np.array(ed["y"], dtype=int)
        proba_dev = model.predict_proba(
            augment_with_squares(pca.transform(X_dev), square_top_k)
        )[:, 1]

        if len(set(y_dev.tolist())) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_dev, proba_dev))

        # train-calibrated threshold (same recipe as fit_lr_gate_pooled.py)
        train_proba = model.predict_proba(Z_tr)[:, 1]
        neg_scores = train_proba[y_tr == 0]
        tau = float(np.max(train_proba))
        for c in np.sort(np.unique(train_proba)):
            if float(np.mean(neg_scores > c)) <= 0.15:
                tau = float(c)
                break

        for meta, p, y in zip(ed["meta"], proba_dev, y_dev):
            row = dict(meta)
            row.update({
                "source": label, "score": float(p), "should_intervene": bool(y),
                "triggered": bool(p > tau),
            })
            all_dev_rows.append(row)

        per_j[j] = {
            "n_train": n, "n_dev": len(y_dev), "pca_dim": nc, "dev_auc": round(auc, 4),
        }

    overall_auc = float("nan")
    overall_metrics = None
    if all_dev_rows:
        y_all = [r["should_intervene"] for r in all_dev_rows]
        s_all = [r["score"] for r in all_dev_rows]
        if len(set(y_all)) >= 2:
            overall_auc = float(roc_auc_score(y_all, s_all))
        overall_metrics = compute_metrics(all_dev_rows)

    return {
        "source": label,
        "per_j": per_j,
        "overall_dev_auc": round(overall_auc, 4) if overall_auc == overall_auc else None,  # NaN check
        "overall_dev_metrics": overall_metrics,
    }, all_dev_rows


def fit_and_eval_layer(
    layer: int, *, hs_root: Path, pca_dim: int, C: float, max_j: int,
    clf: str = "lr", mlp_hidden: int = 8, mlp_alpha: float = 1.0, square_top_k: int = 0,
):
    train_per_kj = collect_split_multilayer(hs_root, "train_pilot", layer)
    dev_per_kj = collect_split_multilayer(hs_root, "dev", layer)
    label = str(layer) if clf == "lr" else f"{layer}_mlp{mlp_hidden}"
    if square_top_k:
        label += f"_sq{square_top_k}"
    result, rows = fit_and_eval_from_per_kj(
        train_per_kj, dev_per_kj, label=label, pca_dim=pca_dim, C=C, max_j=max_j,
        clf=clf, mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha, square_top_k=square_top_k,
    )
    result["layer"] = layer
    return result, rows


def fit_and_eval_concat(
    layers: list[int], *, hs_root: Path, pca_dim: int, C: float, max_j: int,
    clf: str = "lr", mlp_hidden: int = 8, mlp_alpha: float = 1.0, square_top_k: int = 0,
):
    train_per_kj = collect_split_concat(hs_root, "train_pilot", layers)
    dev_per_kj = collect_split_concat(hs_root, "dev", layers)
    label = "concat(" + "+".join(str(l) for l in layers) + ")"
    if clf != "lr":
        label += f"_mlp{mlp_hidden}"
    if square_top_k:
        label += f"_sq{square_top_k}"
    result, rows = fit_and_eval_from_per_kj(
        train_per_kj, dev_per_kj, label=label, pca_dim=pca_dim, C=C, max_j=max_j,
        clf=clf, mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha, square_top_k=square_top_k,
    )
    result["layers"] = layers
    return result, rows


def fit_and_eval_per_j_layers(
    layer_by_j: dict[int, int],
    *,
    hs_root: Path,
    pca_dim: int,
    C: float,
    clf: str = "lr",
    mlp_hidden: int = 8,
    mlp_alpha: float = 1.0,
    square_top_k: int = 0,
):
    """
    Pooled gate training already fits one independent model per transition j
    — nothing requires them all to read the same layer. This mixes in
    whichever layer scored best for each j individually (e.g. j=0,1 -> layer
    15; j=2,3 -> layer 23) instead of picking one layer, or concatenating the
    same fixed set of layers, for every transition uniformly.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    # Cache each distinct layer's collection once, even if it's reused by
    # more than one j — avoids re-reading every npz per transition.
    train_cache: dict[int, dict] = {}
    dev_cache: dict[int, dict] = {}

    def get_train(layer: int) -> dict:
        if layer not in train_cache:
            train_cache[layer] = collect_split_multilayer(hs_root, "train_pilot", layer)
        return train_cache[layer]

    def get_dev(layer: int) -> dict:
        if layer not in dev_cache:
            dev_cache[layer] = collect_split_multilayer(hs_root, "dev", layer)
        return dev_cache[layer]

    per_j: dict[int, dict] = {}
    all_dev_rows: list[dict] = []

    for j, layer in sorted(layer_by_j.items()):
        td = pool_single_j(get_train(layer), j=j)
        ed = pool_single_j(get_dev(layer), j=j)
        if not td["X"] or not ed["X"]:
            per_j[j] = {"skipped": True, "reason": "no rows", "layer": layer}
            continue

        X_tr = np.stack(td["X"], axis=0)
        y_tr = np.array(td["y"], dtype=int)
        n, d = X_tr.shape
        n_pos = int(y_tr.sum())
        if n_pos < 5 or (n - n_pos) < 5:
            per_j[j] = {"skipped": True, "reason": "not enough pos/neg", "n_train": n, "layer": layer}
            continue

        nc = min(pca_dim, n - 1, d)
        pca = PCA(n_components=nc, random_state=0)
        Z_tr = augment_with_squares(pca.fit_transform(X_tr.astype(np.float64)), square_top_k)
        model = make_classifier(clf, C=C, mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha)
        model.fit(Z_tr, y_tr)

        X_dev = np.stack(ed["X"], axis=0).astype(np.float64)
        y_dev = np.array(ed["y"], dtype=int)
        proba_dev = model.predict_proba(
            augment_with_squares(pca.transform(X_dev), square_top_k)
        )[:, 1]
        auc = (
            float(roc_auc_score(y_dev, proba_dev))
            if len(set(y_dev.tolist())) >= 2
            else float("nan")
        )

        train_proba = model.predict_proba(Z_tr)[:, 1]
        neg_scores = train_proba[y_tr == 0]
        tau = float(np.max(train_proba))
        for c in np.sort(np.unique(train_proba)):
            if float(np.mean(neg_scores > c)) <= 0.15:
                tau = float(c)
                break

        for meta, p, y in zip(ed["meta"], proba_dev, y_dev):
            row = dict(meta)
            row.update({
                "source": f"perj_layer{layer}", "score": float(p), "should_intervene": bool(y),
                "triggered": bool(p > tau),
            })
            all_dev_rows.append(row)

        per_j[j] = {
            "layer": layer, "n_train": n, "n_dev": len(y_dev), "pca_dim": nc, "dev_auc": round(auc, 4),
        }

    overall_auc = float("nan")
    overall_metrics = None
    if all_dev_rows:
        y_all = [r["should_intervene"] for r in all_dev_rows]
        s_all = [r["score"] for r in all_dev_rows]
        if len(set(y_all)) >= 2:
            overall_auc = float(roc_auc_score(y_all, s_all))
        overall_metrics = compute_metrics(all_dev_rows)

    label = "perj[" + ",".join(f"{j}:{layer_by_j[j]}" for j in sorted(layer_by_j)) + "]"
    if clf != "lr":
        label += f"_mlp{mlp_hidden}"
    if square_top_k:
        label += f"_sq{square_top_k}"
    return {
        "source": label,
        "per_j": per_j,
        "overall_dev_auc": round(overall_auc, 4) if overall_auc == overall_auc else None,
        "overall_dev_metrics": overall_metrics,
    }, all_dev_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hs-root", type=Path, default=PILOT_HS_ROOT)
    ap.add_argument("--layers", default=None,
                    help="Comma-separated layers to compare. Default: read from run_meta.json.")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-j", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--concat-layers", default=None,
                    help="Comma-separated layers to additionally fit as one concatenated-feature "
                         "PCA+LR (e.g. 15,23), on top of the per-single-layer comparison.")
    ap.add_argument("--skip-single-layers", action="store_true",
                    help="Only run --concat-layers, skip the per-single-layer sweep.")
    ap.add_argument("--classifier", choices=("lr", "mlp"), default="lr",
                    help="Final model on top of PCA features. 'mlp' is a deliberately shallow "
                         "one-hidden-layer, one-activation MLP (see --mlp-hidden/--mlp-alpha), "
                         "not a general deep model — the point is to check whether a *minimal* "
                         "amount of nonlinearity captures anything the linear probe misses.")
    ap.add_argument("--mlp-hidden", type=int, default=8,
                    help="Hidden units in the single hidden layer, only used with --classifier mlp.")
    ap.add_argument("--mlp-alpha", type=float, default=1.0,
                    help="L2 regularization strength for the MLP (MLPClassifier has no "
                         "class_weight='balanced' option like LogisticRegression, so this plus "
                         "early_stopping is what keeps it from just memorizing small pooled-j "
                         "training sets — as few as ~4k rows at j=3).")
    ap.add_argument("--per-j-layers", default=None,
                    help="Mix a different layer per transition instead of one fixed layer/concat "
                         "for all of them, e.g. '0:15,1:15,2:23,3:23' — pooled gate training already "
                         "fits j=0..3 as independent models, so nothing requires them to share a layer.")
    ap.add_argument("--square-top-k", type=int, default=0,
                    help="Append squared terms of the top-k PCA components as extra features before "
                         "LogisticRegression (0 = off). An alternative to --classifier mlp for adding "
                         "a small, interpretable amount of nonlinearity while staying strictly "
                         "LogisticRegression — see augment_with_squares(). Deliberately conservative: "
                         "only a handful of added columns, not a full degree-2 expansion, so a good "
                         "score can't be chalked up to classifier capacity.")
    args = ap.parse_args()

    try:
        import numpy as np  # noqa: F401
        from sklearn.decomposition import PCA  # noqa: F401
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        from sklearn.metrics import roc_auc_score  # noqa: F401
        from tqdm import tqdm
        if args.classifier == "mlp":
            from sklearn.neural_network import MLPClassifier  # noqa: F401
    except ImportError as exc:
        sys.exit(f"Install: pip install numpy scikit-learn joblib tqdm ({exc})")

    run_meta_path = args.hs_root / "run_meta.json"
    if args.layers:
        layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    elif run_meta_path.is_file():
        layers = sorted(json.loads(run_meta_path.read_text(encoding="utf-8"))["layers"])
    else:
        sys.exit(f"No --layers given and no run_meta.json at {run_meta_path}")

    train_dir = args.hs_root / "train_pilot" / "activations"
    dev_dir = args.hs_root / "dev" / "activations"
    if not train_dir.is_dir() or not dev_dir.is_dir():
        sys.exit(
            f"Missing pilot activations under {args.hs_root}\n"
            "Run extract_hidden_states_multilayer_pilot.py first."
        )

    print("=" * 60)
    print("layer comparison pilot")
    print(f"  hs-root: {args.hs_root}")
    print(f"  layers:  {layers}")
    print("=" * 60)

    results: dict[str, dict] = {}
    row_order: list[str] = []

    if not args.skip_single_layers:
        for layer in tqdm(layers, desc="layers", unit="layer"):
            res, _ = fit_and_eval_layer(
                layer, hs_root=args.hs_root, pca_dim=args.pca_dim, C=args.C, max_j=args.max_j,
                clf=args.classifier, mlp_hidden=args.mlp_hidden, mlp_alpha=args.mlp_alpha,
                square_top_k=args.square_top_k,
            )
            results[res["source"]] = res
            row_order.append(res["source"])

    if args.concat_layers:
        # Not deduped/sorted into a set: repeating a layer (e.g. "15,15") is a
        # deliberately supported control — it tests whether concatenation
        # helps because of genuinely complementary information between two
        # *different* layers, or just because PCA gets more raw input
        # dimensions to work with regardless of what's in them.
        concat_layers = [int(x) for x in args.concat_layers.split(",") if x.strip()]
        print(f"\nFitting concat({'+'.join(map(str, concat_layers))}) ...")
        res, _ = fit_and_eval_concat(
            concat_layers, hs_root=args.hs_root, pca_dim=args.pca_dim, C=args.C, max_j=args.max_j,
            clf=args.classifier, mlp_hidden=args.mlp_hidden, mlp_alpha=args.mlp_alpha,
            square_top_k=args.square_top_k,
        )
        results[res["source"]] = res
        row_order.append(res["source"])

    if args.per_j_layers:
        layer_by_j: dict[int, int] = {}
        for part in args.per_j_layers.split(","):
            part = part.strip()
            if not part:
                continue
            j_str, layer_str = part.split(":")
            layer_by_j[int(j_str)] = int(layer_str)
        print(f"\nFitting per-j mix {sorted(layer_by_j.items())} ...")
        res, _ = fit_and_eval_per_j_layers(
            layer_by_j, hs_root=args.hs_root, pca_dim=args.pca_dim, C=args.C,
            clf=args.classifier, mlp_hidden=args.mlp_hidden, mlp_alpha=args.mlp_alpha,
            square_top_k=args.square_top_k,
        )
        results[res["source"]] = res
        row_order.append(res["source"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "layer_compare_metrics.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    j_values = sorted({j for r in results.values() for j in r["per_j"].keys()})
    header = ["source"] + [f"j={j}({pooled_transition_label(j)})" for j in j_values] + ["overall"]
    print("\n" + " | ".join(f"{h:<16}" for h in header))
    print("-" * (19 * len(header)))
    for key in row_order:
        r = results[key]
        cells = [f"{key:<16}"]
        for j in j_values:
            pj = r["per_j"].get(j)
            auc = pj.get("dev_auc") if pj and not pj.get("skipped") else None
            cells.append(f"{auc if auc is not None else 'n/a':<16}")
        cells.append(f"{r['overall_dev_auc']}")
        print(" | ".join(str(c) for c in cells))

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
