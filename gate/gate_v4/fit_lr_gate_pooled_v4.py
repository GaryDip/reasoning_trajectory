#!/usr/bin/env python3
"""
Gate v4 (bilinear transition model, replaces the earlier MLP attempt -- see conversation:
the MLP on concat(PCA(h_after), PCA(Delta_j)) overfit badly at width 32 and still trailed
v3 even after shrinking to width 8 with heavy L2, so this abandons "let an unconstrained
net find the interaction" in favor of a small model structurally built around ONE specific
hypothesis: v3's LR on concat(h_after, Delta_j) can only score h_after and Delta_j
independently and add them -- it cannot express "the same Delta_j means something different
depending on where h_{j-1} started."

Model, for each candidate i at pooled transition j:

    z_pre  = W_pre  * h_{j-1}          # low-rank projection of the state BEFORE this hop
    z_post = W_post * h_{j,i}          # low-rank projection of the state AFTER candidate i
    score_i = (z_pre . z_post) + w_delta^T * (h_{j,i} - h_{j-1}) + b

The dot product z_pre . z_post is a low-rank bilinear form (equivalent to
h_{j-1}^T (W_pre^T W_post) h_{j,i}, a rank-<=r matrix instead of the full d x d one) --
this is what actually captures "is this specific post-state compatible with this specific
pre-state," not just "do h_after and Delta_j individually look normal." The delta term
keeps v2's original local signal available on top of it. W_pre/W_post/w_delta are all
learned end-to-end via BCE against the SAME wrong-hop label v2/v3 use -- no PCA
preprocessing (the low-rank projections themselves do the dimensionality reduction).

No new Llama extraction needed -- reuses h_{j-1} (hidden[j]) and h_{j,i} (hidden[j+1]) from
hidden_states/pilot_multilayer/, the same source v2/v3 already read from, for BOTH gold and
counterfactual rows.

First real run (rank=64) beat every prior model on AUC (0.9457 vs v3's 0.9357) but trailed
v3 on F1 (0.7491 vs 0.793) -- same failure mode the MLP attempt hit: the threshold picked
from the FULL train set's own score distribution (in-sample, optimistic) doesn't transfer
to dev, where FPR overshoots the 0.15 target by 2x. This version fixes that by picking the
threshold from out-of-fold predictions (--tau-cv-folds, default 5): K-fold cross-validation
over the training data, each fold's held-out rows scored by a model that never trained on
them, threshold picked on the POOLED out-of-fold scores -- an honest estimate of how the
deployed (full-data-trained) model's scores will actually distribute on unseen data, instead
of an in-sample one.

Usage:
  python fit_lr_gate_pooled_v4.py
  python fit_lr_gate_pooled_v4.py --rank 32 --tau-cv-folds 5
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
RESULTS_DIR = HERE / "results_pooled_v4"
ARTIFACTS_DIR = HERE / "artifacts_pooled_v4"

sys.path.insert(0, str(GATE_DIR))

from fit_lr_gate import load_manifest_meta, read_hidden_from_npz, read_K_from_npz  # noqa: E402
from hop_labels import err_hop_bucket_for_row, parse_wrong_hops, should_intervene_at_j  # noqa: E402

DEFAULT_PER_J_LAYERS = {0: 15, 1: 15, 2: 23, 3: 23}


def pooled_transition_label(j: int) -> str:
    if j == 0:
        return "Q->E1"
    return f"E{j}->E{j + 1}"


def collect_split_v4(hs_root: Path, split: str, *, layer: int) -> dict[tuple[int, int], dict]:
    """Like v3's collect_split, but ALSO keeps h_before (=hidden[j], the state right before
    this hop) -- v3 only kept h_after and delta, discarding h_before once delta was computed,
    but the bilinear term here needs h_before on its own, not just folded into a difference."""
    manifest = load_manifest_meta(hs_root / split / "manifest.jsonl")
    data: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"X_before": [], "X_after": [], "y": [], "meta": []}
    )

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
                h_before = hidden[j].astype(np.float32)
                h_after = hidden[j + 1].astype(np.float32)
                label = should_intervene_at_j(wrong_hops, j, trace_type=trace_type)
                data[(K, j)]["X_before"].append(h_before)
                data[(K, j)]["X_after"].append(h_after)
                data[(K, j)]["y"].append(label)
                data[(K, j)]["meta"].append({
                    "id": eid, "trace_type": trace_type, "wrong_hops": wrong_hops, "K": K, "j": j,
                })
    return data


def collect_split_pooled_v4(hs_root: Path, split: str, *, max_j: int, layer_by_j: dict[int, int]) -> dict[int, dict]:
    pooled: dict[int, dict] = defaultdict(lambda: {"X_before": [], "X_after": [], "y": [], "meta": []})
    cache: dict[int, dict[tuple[int, int], dict]] = {}
    for j, lyr in sorted(layer_by_j.items()):
        if j > max_j:
            continue
        if lyr not in cache:
            cache[lyr] = collect_split_v4(hs_root, split, layer=lyr)
        per_kj = cache[lyr]
        for (K, jj), group in sorted(per_kj.items()):
            if jj != j or jj >= K:
                continue
            for h_before, h_after, label, meta in zip(
                group["X_before"], group["X_after"], group["y"], group["meta"]
            ):
                pooled[j]["X_before"].append(h_before)
                pooled[j]["X_after"].append(h_after)
                pooled[j]["y"].append(label)
                pooled[j]["meta"].append(dict(meta))
    return pooled


def group_split(ids: list[str], *, holdout_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Group-aware holdout split (by example id) for early stopping only."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(ids)
    rng.shuffle(uniq)
    n_holdout = max(1, int(len(uniq) * holdout_frac))
    holdout_ids = set(uniq[:n_holdout].tolist())
    idx = np.arange(len(ids))
    is_holdout = np.array([ids[i] in holdout_ids for i in idx])
    return idx[~is_holdout], idx[is_holdout]


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


def pick_threshold(scores: np.ndarray, y: np.ndarray, *, target_fpr: float) -> float:
    neg_scores = scores[y == 0]
    tau = float(np.max(scores))
    for c in np.sort(np.unique(scores)):
        if float(np.mean(neg_scores > c)) <= target_fpr:
            return float(c)
    return tau


def make_model(in_dim: int, rank: int, device: str, seed: int):
    import torch
    import torch.nn as nn

    class BilinearTransitionGate(nn.Module):
        def __init__(self, in_dim: int, rank: int):
            super().__init__()
            self.w_pre = nn.Linear(in_dim, rank, bias=False)
            self.w_post = nn.Linear(in_dim, rank, bias=False)
            self.w_delta = nn.Linear(in_dim, 1, bias=False)
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, h_prev, h_cur):
            pre = self.w_pre(h_prev)
            post = self.w_post(h_cur)
            interaction = (pre * post).sum(dim=-1)
            transition = self.w_delta(h_cur - h_prev).squeeze(-1)
            return interaction + transition + self.bias

    torch.manual_seed(seed)
    return BilinearTransitionGate(in_dim=in_dim, rank=rank).to(device)


def train_one_model(
    X_before: np.ndarray, X_after: np.ndarray, y: np.ndarray, ids: list[str], *,
    rank: int, weight_decay: float, lr: float, epochs: int, batch_size: int, patience: int,
    holdout_frac: float, seed: int, device: str,
):
    import torch
    import torch.nn as nn

    n = len(y)
    n_pos = int(y.sum())
    fit_idx, holdout_idx = group_split(ids, holdout_frac=holdout_frac, seed=seed)
    pos_weight = torch.tensor([(n - n_pos) / max(n_pos, 1)], dtype=torch.float32, device=device)

    model = make_model(X_before.shape[1], rank, device, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xb_fit = torch.tensor(X_before[fit_idx], device=device)
    Xa_fit = torch.tensor(X_after[fit_idx], device=device)
    y_fit = torch.tensor(y[fit_idx], device=device)
    Xb_hold = torch.tensor(X_before[holdout_idx], device=device)
    Xa_hold = torch.tensor(X_after[holdout_idx], device=device)
    y_hold = torch.tensor(y[holdout_idx], device=device)

    best_state, best_loss, bad_epochs = None, float("inf"), 0
    rng = np.random.default_rng(seed)
    n_fit = len(fit_idx)
    for _epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_fit)
        for start in range(0, n_fit, batch_size):
            batch = perm[start : start + batch_size]
            opt.zero_grad()
            logits = model(Xb_fit[batch], Xa_fit[batch])
            loss = loss_fn(logits, y_fit[batch])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            hold_logits = model(Xb_hold, Xa_hold)
            hold_loss = nn.functional.binary_cross_entropy_with_logits(
                hold_logits, y_hold, pos_weight=pos_weight
            ).item()
        if hold_loss < best_loss - 1e-5:
            best_loss = hold_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


def cv_out_of_fold_proba(
    X_before: np.ndarray, X_after: np.ndarray, y: np.ndarray, ids: list[str], *,
    n_folds: int, rank: float, weight_decay: float, lr: float, epochs: int, batch_size: int,
    patience: int, holdout_frac: float, seed: int, device: str,
) -> np.ndarray:
    """K-fold CV purely to get an honest (out-of-fold) probability for every training row,
    used only to pick the deployment threshold -- NOT the model that gets deployed (that's
    trained once on the full data in main(), same as before)."""
    import torch
    from sklearn.model_selection import StratifiedKFold

    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_before, y)):
        model, _ = train_one_model(
            X_before[tr_idx], X_after[tr_idx], y[tr_idx], [ids[i] for i in tr_idx],
            rank=rank, weight_decay=weight_decay, lr=lr, epochs=epochs, batch_size=batch_size,
            patience=patience, holdout_frac=holdout_frac, seed=seed + fold_i, device=device,
        )
        with torch.no_grad():
            logits = model(
                torch.tensor(X_before[te_idx], device=device), torch.tensor(X_after[te_idx], device=device)
            ).cpu().numpy()
        oof[te_idx] = 1.0 / (1.0 + np.exp(-logits))
    return oof


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hs-root", type=Path, default=HS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    ap.add_argument("--rank", type=int, default=64, help="Low-rank dim r for W_pre/W_post.")
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--tau-cv-folds", type=int, default=5,
                     help="K-fold CV to pick an out-of-fold-calibrated threshold instead of "
                          "an in-sample one (see module docstring). 0 disables (old behavior).")
    ap.add_argument("--split-eval", choices=("dev", "train"), default="dev")
    ap.add_argument("--target-fpr", type=float, default=0.15)
    ap.add_argument("--max-j", type=int, default=3)
    ap.add_argument("--train-split-name", default="train_pilot")
    ap.add_argument("--per-j-layers", default="0:15,1:15,2:23,3:23",
                     help="Same per-j layer choice gate v2/v3 already validated.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
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
        import torch
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        sys.exit(f"Install: pip install numpy scikit-learn torch tqdm ({exc})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print("=" * 60)
    print("gate v4: bilinear(h_{j-1}, h_j) + linear(Delta_j) -- state-transition compatibility")
    print(f"  hs-root:       {args.hs_root}")
    print(f"  artifacts-dir: {args.artifacts_dir}")
    print(f"  split-eval:    {args.split_eval}")
    print(f"  per-j layers:  {sorted(layer_by_j.items())}")
    print(f"  rank:          {args.rank}   weight_decay={args.weight_decay}")
    print(f"  tau-cv-folds:  {args.tau_cv_folds}")
    print("=" * 60)

    print("\nLoading train split ...", flush=True)
    train_data = collect_split_pooled_v4(args.hs_root, args.train_split_name, max_j=args.max_j, layer_by_j=layer_by_j)
    print(f"  pooled j groups: {len(train_data)}")

    print(f"Loading {args.split_eval} split ...", flush=True)
    eval_data = collect_split_pooled_v4(args.hs_root, args.split_eval, max_j=args.max_j, layer_by_j=layer_by_j)
    print(f"  pooled j groups: {len(eval_data)}")

    print("\nFitting pooled bilinear transition model ...")
    models: dict[int, dict] = {}
    for j, td in sorted(train_data.items()):
        X_before = np.stack(td["X_before"], axis=0).astype(np.float32)
        X_after = np.stack(td["X_after"], axis=0).astype(np.float32)
        y = np.array(td["y"], dtype=np.float32)
        ids = [m["id"] for m in td["meta"]]
        n = len(y)
        n_pos = int(y.sum())
        K_used = sorted({int(m["K"]) for m in td["meta"]})
        if n_pos < 5 or (n - n_pos) < 5:
            continue

        common_kwargs = dict(
            rank=args.rank, weight_decay=args.weight_decay, lr=args.lr, epochs=args.epochs,
            batch_size=args.batch_size, patience=args.patience, holdout_frac=args.holdout_frac,
            seed=args.seed, device=device,
        )

        if args.tau_cv_folds > 1:
            oof_proba = cv_out_of_fold_proba(
                X_before, X_after, y, ids, n_folds=args.tau_cv_folds, **common_kwargs,
            )
            tau = pick_threshold(oof_proba, y, target_fpr=args.target_fpr)
        else:
            oof_proba = None

        model, best_loss = train_one_model(X_before, X_after, y, ids, **common_kwargs)

        with torch.no_grad():
            train_logits = model(
                torch.tensor(X_before, device=device), torch.tensor(X_after, device=device)
            ).cpu().numpy()
        train_proba = 1.0 / (1.0 + np.exp(-train_logits))

        if oof_proba is None:
            tau = pick_threshold(train_proba, y, target_fpr=args.target_fpr)

        train_auc = round(float(roc_auc_score(y, train_proba)), 4)
        oof_auc = round(float(roc_auc_score(y, oof_proba)), 4) if oof_proba is not None else None
        models[j] = {
            "model": model, "threshold": tau, "train_auc": train_auc,
            "layer": layer_by_j[j], "K_used": K_used,
        }
        oof_str = f" oof_auc={oof_auc}" if oof_auc is not None else ""
        print(f"  j={j} {pooled_transition_label(j):>10} K={K_used} n={n} pos={100*n_pos/n:.1f}% "
              f"tau={tau:.3f} train_auc={train_auc}{oof_str} best_holdout_loss={best_loss:.4f}")

    if not args.no_save_artifacts:
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        meta_rows = []
        for j, m in models.items():
            fname = f"j{j}.pt"
            torch.save(m["model"].state_dict(), args.artifacts_dir / fname)
            meta_rows.append({
                "j": j, "transition": pooled_transition_label(j), "layer": m["layer"],
                "K_used": m["K_used"], "rank": args.rank,
                "threshold": m["threshold"], "train_auc": m["train_auc"], "file": fname,
            })
        meta = {
            "pooling": "semantic_j_across_K_excluding_final",
            "feature": "bilinear(W_pre h_{j-1}, W_post h_j) + linear(Delta_j) -- state-"
                       "transition compatibility, trained end-to-end via BCE, threshold "
                       "picked via out-of-fold CV (see module docstring)",
            "n_models": len(models),
            "rank": args.rank, "weight_decay": args.weight_decay,
            "tau_cv_folds": args.tau_cv_folds,
            "target_fpr": args.target_fpr,
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
        Xb_ev = torch.tensor(np.stack(ed["X_before"], axis=0).astype(np.float32), device=device)
        Xa_ev = torch.tensor(np.stack(ed["X_after"], axis=0).astype(np.float32), device=device)
        with torch.no_grad():
            logits = m["model"](Xb_ev, Xa_ev).cpu().numpy()
        proba = 1.0 / (1.0 + np.exp(-logits))
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
    out_path = args.out_dir / f"{args.split_eval}_lr_pooled_v4_metrics.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\noverall: {results['overall']}  auc={results['auc']}")
    print("by_j:")
    for j, m in results["by_j"].items():
        print(f"  j={j}: {m}")
    print(f"\nSaved -> {out_path}")
    print("\nCompare against gate v3 (concat + LR): gate/gate_v3/results_pooled_v3/dev_lr_pooled_v3_metrics.json")
    print("Compare against gate v2 (Delta-only):   gate/results_pooled_v2/dev_lr_pooled_metrics.json")


if __name__ == "__main__":
    main()
