#!/usr/bin/env python3
"""
Train a FRESH probe on the raw final-prefix hidden state h_K (not a delta),
and check whether its held-out score rises with n_wrong (0 = gold, 1..K =
number of corrupted hops). This is deliberately NOT the existing
gate/artifacts_pooled_v2 (trained on Delta_j = h_j - h_{j-1}, answering "did
THIS step just get tampered with relative to the last one") — that's a
local, step-to-step signal, not a "how good does this whole prefix look"
signal. Beam search needs the latter (comparing whole candidate paths
against each other), so the probe here scores h_K directly.

Two modes:
  --train-hidden-dir + --dev-hidden-dir: fit PCA+LR on ALL of train, then
  report the (K, n_wrong) -> score trend purely on dev — a genuine
  cross-split generalization check (MuSiQue's own official train/dev split,
  fully disjoint questions), not just an internal random split of one split.
  This is what you want once you actually have both extracted.

  --hidden-dir (single, original behavior): internal 75/25 split by
  source_id within that one directory — a quick sanity check when you've
  only extracted one split so far (e.g. the dev-only smoke test earlier).

Splits by source_id (not by row) whenever a split needs to happen at all,
since build_multi_error_traces.py emits n_wrong=0..K variants of the SAME
question — a random row split would leak near-identical prefixes across
train/test.

Usage:
  python score_and_analyze.py --train-hidden-dir hidden_states_full/train \
      --dev-hidden-dir hidden_states_full/dev
  python score_and_analyze.py --hidden-dir hidden_states_full/dev
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_manifest_and_hidden(hidden_dir: Path):
    manifest_path = hidden_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        sys.exit(f"Missing manifest: {manifest_path}")

    rows = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    X, K_list, n_wrong_list, source_ids = [], [], [], []
    n_skipped = 0
    for row in rows:
        K = int(row["K"])
        npz_path = hidden_dir / row["path"]
        if not npz_path.is_file() or row["num_prefixes_saved"] < K + 1:
            n_skipped += 1
            continue
        hidden = np.load(npz_path)["hidden"]
        X.append(hidden[K])  # raw final-prefix state, NOT a delta
        K_list.append(K)
        n_wrong_list.append(int(row.get("n_wrong", len(row.get("wrong_hops") or []))))
        source_ids.append(str(row.get("source_id") or row["id"]))

    if n_skipped:
        print(f"Skipped {n_skipped} rows (missing npz or incomplete prefixes)")
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(K_list, dtype=np.int64),
        np.asarray(n_wrong_list, dtype=np.int64),
        np.asarray(source_ids),
    )


def load_all_prefix_positions(hidden_dir: Path):
    """Like load_manifest_and_hidden, but returns EVERY intermediate prefix h_1..h_K of every
    trace (not just the final h_K) — the manifest already has num_prefixes_saved >= K+1 rows per
    trace since extract_full_hidden_states.py always takes the full trajectory. For a K=4 trace,
    the label at position j is NOT the trace's overall n_wrong: h_j has only ever attended over
    hops 1..j (hop j+1..K text hasn't been generated yet at that point), so the right label is
    how many of wrong_hops occurred at or before j.
    """
    manifest_path = hidden_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        sys.exit(f"Missing manifest: {manifest_path}")

    rows = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    X, K_list, j_list, n_eff_list, source_ids = [], [], [], [], []
    n_skipped = 0
    for row in rows:
        K = int(row["K"])
        npz_path = hidden_dir / row["path"]
        if not npz_path.is_file() or row["num_prefixes_saved"] < K + 1:
            n_skipped += 1
            continue
        wrong_hops = set(int(h) for h in (row.get("wrong_hops") or []))
        hidden = np.load(npz_path)["hidden"]
        for j in range(1, K + 1):
            X.append(hidden[j])
            K_list.append(K)
            j_list.append(j)
            n_eff_list.append(sum(1 for h in wrong_hops if h <= j))
            source_ids.append(str(row.get("source_id") or row["id"]))

    if n_skipped:
        print(f"Skipped {n_skipped} rows (missing npz or incomplete prefixes)")
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(K_list, dtype=np.int64),
        np.asarray(j_list, dtype=np.int64),
        np.asarray(n_eff_list, dtype=np.int64),
        np.asarray(source_ids),
    )


def report_position_trend(K_arr: np.ndarray, j_arr: np.ndarray, n_eff_arr: np.ndarray,
                           scores: np.ndarray, *, label: str) -> None:
    """Checks the score trend at INTERMEDIATE positions too — e.g. does h_2 inside a 4-hop trace
    (which has only seen hops 1-2 so far) behave like h_j elsewhere, as a function of how many of
    ITS OWN preceding hops were corrupted (not the whole trace's final n_wrong)."""
    by_kj_neff: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    by_j_neff: dict[tuple[int, int], list[float]] = defaultdict(list)
    for K, j, n_eff, score in zip(K_arr, j_arr, n_eff_arr, scores):
        by_kj_neff[(int(K), int(j), int(n_eff))].append(float(score))
        by_j_neff[(int(j), int(n_eff))].append(float(score))

    print(f"\n[{label}] Per (K_total, position j, n_wrong up-to-j), score at h_j:")
    print(f"{'K':>3} {'j':>3} {'n_eff':>6} {'n':>6} {'mean':>8} {'median':>8}")
    for (K, j, n_eff), s in sorted(by_kj_neff.items()):
        arr = np.array(s)
        print(f"{K:>3} {j:>3} {n_eff:>6} {len(arr):>6} {arr.mean():>8.4f} {np.median(arr):>8.4f}")

    print(f"\n[{label}] Pooled ACROSS K_total, by (position j, n_wrong up-to-j) — "
          f"e.g. j=2 rows here mix K=2's final hop with K=3/K=4's intermediate hop 2:")
    print(f"{'j':>3} {'n_eff':>6} {'n':>6} {'mean':>8} {'median':>8}")
    for (j, n_eff), s in sorted(by_j_neff.items()):
        arr = np.array(s)
        print(f"{j:>3} {n_eff:>6} {len(arr):>6} {arr.mean():>8.4f} {np.median(arr):>8.4f}")

    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(n_eff_arr, scores)
        print(f"\n[{label}] Spearman(n_wrong up-to-j, score), pooled over ALL positions "
              f"= {rho:.4f} (p={pval:.2e}), n={len(scores)}")
    except ImportError:
        pass


class MLPScorer:
    """Small MLP that regresses the CONTINUOUS severity ratio n_eff/j directly (no binary
    classification step) — trades LR's simplicity for letting the model exploit the full
    graded label (0..j corrupted hops) instead of collapsing it to corrupted-vs-not."""

    def __init__(self, in_dim: int, hidden: list[int]):
        import torch.nn as nn

        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]  # bounded to [0,1], same range as n_eff/j
        self.net = nn.Sequential(*layers)

    def to(self, device):
        self.net.to(device)
        return self

    def parameters(self):
        return self.net.parameters()

    def train_mode(self):
        self.net.train()

    def eval_mode(self):
        self.net.eval()

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


def train_mlp_probe(
    Z_train: np.ndarray, ratio_train: np.ndarray, Z_holdout: np.ndarray, ratio_holdout: np.ndarray,
    *, hidden: list[int], epochs: int, lr: float, batch_size: int, seed: int,
):
    """Fit MLPScorer by MSE regression against ratio = n_eff/j (in [0,1]), with early
    stopping on a held-out TRAIN subset (never dev — dev stays untouched during fitting,
    same principle as the LR path)."""
    import torch

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPScorer(Z_train.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X_tr = torch.tensor(Z_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(ratio_train, dtype=torch.float32, device=device)
    X_ho = torch.tensor(Z_holdout, dtype=torch.float32, device=device)
    y_ho = torch.tensor(ratio_holdout, dtype=torch.float32, device=device)

    n = X_tr.shape[0]
    best_state, best_loss = None, float("inf")
    patience, bad_epochs = 5, 0
    for epoch in range(1, epochs + 1):
        model.train_mode()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            pred = model.forward(X_tr[idx])
            loss = torch.nn.functional.mse_loss(pred, y_tr[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        model.eval_mode()
        with torch.no_grad():
            ho_loss = torch.nn.functional.mse_loss(model.forward(X_ho), y_ho).item()
        print(f"  epoch {epoch:>3}: train_mse={total_loss / n:.5f}  holdout_mse={ho_loss:.5f}")

        if ho_loss < best_loss - 1e-6:
            best_loss, best_state, bad_epochs = ho_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {epoch} (best holdout_mse={best_loss:.5f})")
                break

    model.load_state_dict(best_state)
    model.eval_mode()
    return model


def mlp_predict(model, Z: np.ndarray) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        return model.forward(torch.tensor(Z, dtype=torch.float32, device=device)).cpu().numpy()


def group_train_test_split(source_ids: np.ndarray, *, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(source_ids)
    rng.shuffle(unique_groups)
    n_test_groups = max(1, int(len(unique_groups) * test_frac))
    test_groups = set(unique_groups[:n_test_groups].tolist())
    is_test = np.array([g in test_groups for g in source_ids])
    return ~is_test, is_test


def report_trend(K_eval: np.ndarray, n_wrong_eval: np.ndarray, scores: np.ndarray, *, label: str) -> None:
    by_k_nwrong: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_nwrong: dict[int, list[float]] = defaultdict(list)
    for K, n_wrong, score in zip(K_eval, n_wrong_eval, scores):
        by_k_nwrong[(int(K), int(n_wrong))].append(float(score))
        by_nwrong[int(n_wrong)].append(float(score))

    print(f"\n[{label}] Per (K, n_wrong), fresh probe score on raw h_K:")
    print(f"{'K':>3} {'n_wrong':>8} {'n':>6} {'mean':>8} {'median':>8}")
    for (K, n_wrong), s in sorted(by_k_nwrong.items()):
        arr = np.array(s)
        print(f"{K:>3} {n_wrong:>8} {len(arr):>6} {arr.mean():>8.4f} {np.median(arr):>8.4f}")

    print(f"\n[{label}] Pooled over K, by n_wrong:")
    print(f"{'n_wrong':>8} {'n':>6} {'mean':>8} {'median':>8} {'std':>8}")
    for n_wrong, s in sorted(by_nwrong.items()):
        arr = np.array(s)
        print(f"{n_wrong:>8} {len(arr):>6} {arr.mean():>8.4f} {np.median(arr):>8.4f} {arr.std():>8.4f}")

    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(n_wrong_eval, scores)
        print(f"\n[{label}] Spearman(n_wrong, probe score) = {rho:.4f} (p={pval:.2e}), n={len(scores)}")
    except ImportError:
        print(f"\n({label}: scipy not installed - skipping Spearman correlation; table above already shows the trend)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden-dir", type=Path, default=None,
                     help="Single-directory mode: internal 75/25 split within this one split.")
    ap.add_argument("--train-hidden-dir", type=Path, default=None,
                     help="Cross-split mode: fit on ALL of this (e.g. hidden_states_full/train).")
    ap.add_argument("--dev-hidden-dir", type=Path, default=None,
                     help="Cross-split mode: report the trend purely on ALL of this "
                          "(e.g. hidden_states_full/dev) — never touched during fitting.")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--test-frac", type=float, default=0.25,
                     help="Only used in single-directory mode.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="If set, save pca.joblib/model.pt/meta.json here (cross-split mode only).")
    ap.add_argument("--skip-position-check", action="store_true",
                     help="Skip scoring every intermediate prefix (h_1..h_K, not just the final "
                          "h_K) with the same fitted probe — e.g. whether h_2 inside a 4-hop trace "
                          "tracks how many of ITS OWN first 2 hops were corrupted. On by default "
                          "since it's free (no extra extraction, same npz files already have every "
                          "prefix); pass this to skip it and only reproduce the original final-h_K-"
                          "only report.")
    ap.add_argument("--pool-positions", action="store_true",
                     help="Fit PCA+LR on EVERY prefix h_1..h_K pooled from train (label = whether "
                          "any of ITS OWN preceding hops were corrupted so far), not just the final "
                          "h_K — so the probe is trained to score partial trajectories directly, "
                          "not just zero-shot-generalized to them from an endpoints-only fit. Each "
                          "trace contributes K training rows instead of 1, which also help the "
                          "underrepresented K=3/K=4 buckets. Without this flag, fitting is on final "
                          "h_K only (original behavior) and every-position scoring (unless "
                          "--skip-position-check) is a generalization check of that same model.")
    args = ap.parse_args()

    cross_split = args.train_hidden_dir is not None or args.dev_hidden_dir is not None
    if cross_split and (args.train_hidden_dir is None or args.dev_hidden_dir is None):
        sys.exit("Cross-split mode needs BOTH --train-hidden-dir and --dev-hidden-dir.")
    if not cross_split and args.hidden_dir is None:
        sys.exit("Pass either --hidden-dir, or both --train-hidden-dir and --dev-hidden-dir.")

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    if cross_split and args.pool_positions:
        X_train, K_train, j_train, n_eff_train, _src_train = load_all_prefix_positions(args.train_hidden_dir)
        X_dev, K_dev, j_dev, n_eff_dev, _src_dev = load_all_prefix_positions(args.dev_hidden_dir)
        print(f"Train ({args.train_hidden_dir}): {len(X_train)} rows, ALL positions pooled")
        print(f"Dev   ({args.dev_hidden_dir}): {len(X_dev)} rows, ALL positions pooled")
        if len(X_train) == 0 or len(X_dev) == 0:
            sys.exit("No usable rows found in train or dev.")

        y_train = (n_eff_train > 0).astype(int)
        if y_train.sum() == 0 or (1 - y_train).sum() == 0:
            sys.exit("Train split has only one class.")

        pca = PCA(n_components=min(args.pca_dim, X_train.shape[0] - 1, X_train.shape[1]))
        Z_train = pca.fit_transform(X_train)
        Z_dev = pca.transform(X_dev)

        lr = LogisticRegression(max_iter=2000)
        lr.fit(Z_train, y_train)
        print(f"Train accuracy (corrupted-so-far vs not, ALL positions pooled): {lr.score(Z_train, y_train):.4f}")

        dev_scores = lr.predict_proba(Z_dev)[:, 1]
        final_mask = j_dev == K_dev
        report_trend(K_dev[final_mask], n_eff_dev[final_mask], dev_scores[final_mask],
                      label="DEV final h_K (probe trained on ALL positions pooled)")
        report_position_trend(K_dev, j_dev, n_eff_dev, dev_scores,
                               label="DEV every position (probe trained on ALL positions pooled)")

        if args.out_dir is not None:
            import joblib
            args.out_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(pca, args.out_dir / "pca.joblib")
            joblib.dump(lr, args.out_dir / "lr.joblib")
            meta = {"pca_dim": args.pca_dim, "n_train": len(X_train), "n_dev": len(X_dev),
                     "pool_positions": True}
            (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"\nSaved pca.joblib/lr.joblib/meta.json -> {args.out_dir}")
        return

    if cross_split:
        X_train_raw, K_train, n_wrong_train, _src_train = load_manifest_and_hidden(args.train_hidden_dir)
        X_dev_raw, K_dev, n_wrong_dev, _src_dev = load_manifest_and_hidden(args.dev_hidden_dir)
        print(f"Train ({args.train_hidden_dir}): {len(X_train_raw)} prefixes")
        print(f"Dev   ({args.dev_hidden_dir}): {len(X_dev_raw)} prefixes")
        if len(X_train_raw) == 0 or len(X_dev_raw) == 0:
            sys.exit("No usable rows found in train or dev.")

        y_train = (n_wrong_train > 0).astype(int)
        if y_train.sum() == 0 or (1 - y_train).sum() == 0:
            sys.exit("Train split has only one class.")

        pca = PCA(n_components=min(args.pca_dim, X_train_raw.shape[0] - 1, X_train_raw.shape[1]))
        Z_train = pca.fit_transform(X_train_raw)
        Z_dev = pca.transform(X_dev_raw)

        lr = LogisticRegression(max_iter=2000)
        lr.fit(Z_train, y_train)
        print(f"Train accuracy (corrupted vs not): {lr.score(Z_train, y_train):.4f}")

        dev_scores = lr.predict_proba(Z_dev)[:, 1]
        report_trend(K_dev, n_wrong_dev, dev_scores, label="DEV, never seen during fitting")

        if not args.skip_position_check:
            X_dev_pos, K_dev_pos, j_dev_pos, n_eff_dev_pos, _src_dev_pos = load_all_prefix_positions(
                args.dev_hidden_dir
            )
            print(f"\nDev, ALL intermediate prefixes (not just final h_K): {len(X_dev_pos)} rows")
            Z_dev_pos = pca.transform(X_dev_pos)
            dev_pos_scores = lr.predict_proba(Z_dev_pos)[:, 1]
            report_position_trend(K_dev_pos, j_dev_pos, n_eff_dev_pos, dev_pos_scores,
                                   label="DEV, same probe scored at every prefix position")

        if args.out_dir is not None:
            import joblib
            import torch
            args.out_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(pca, args.out_dir / "pca.joblib")
            joblib.dump(lr, args.out_dir / "lr.joblib")
            meta = {"pca_dim": args.pca_dim, "n_train": len(X_train_raw), "n_dev": len(X_dev_raw)}
            (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"\nSaved pca.joblib/lr.joblib/meta.json -> {args.out_dir}")
        return

    if args.pool_positions:
        X, K_arr, j_arr, n_eff_arr, source_ids = load_all_prefix_positions(args.hidden_dir)
        if len(X) == 0:
            sys.exit("No usable rows found.")
        print(f"Loaded {len(X)} rows ({len(np.unique(source_ids))} distinct questions), ALL positions pooled")

        train_mask, test_mask = group_train_test_split(source_ids, test_frac=args.test_frac, seed=args.seed)
        y = (n_eff_arr > 0).astype(int)
        if y[train_mask].sum() == 0 or (1 - y[train_mask]).sum() == 0:
            sys.exit("Train split has only one class — increase sample size / --test-frac.")

        pca = PCA(n_components=min(args.pca_dim, train_mask.sum() - 1, X.shape[1]))
        Z_train = pca.fit_transform(X[train_mask])
        Z_test = pca.transform(X[test_mask])

        lr = LogisticRegression(max_iter=2000)
        lr.fit(Z_train, y[train_mask])
        train_acc = lr.score(Z_train, y[train_mask])
        test_scores = lr.predict_proba(Z_test)[:, 1]
        print(f"Train accuracy (corrupted-so-far vs not, ALL positions pooled): {train_acc:.4f}")
        print(f"Train n={train_mask.sum()}, test n={test_mask.sum()}")

        K_test, j_test, n_eff_test = K_arr[test_mask], j_arr[test_mask], n_eff_arr[test_mask]
        final_sub_mask = j_test == K_test
        report_trend(K_test[final_sub_mask], n_eff_test[final_sub_mask], test_scores[final_sub_mask],
                      label="HELD-OUT final h_K (probe trained on ALL positions pooled)")
        report_position_trend(K_test, j_test, n_eff_test, test_scores,
                               label="HELD-OUT every position (probe trained on ALL positions pooled)")
        return

    # Single-directory mode (original behavior): internal split within one split.
    X, K_arr, n_wrong_arr, source_ids = load_manifest_and_hidden(args.hidden_dir)
    if len(X) == 0:
        sys.exit("No usable rows found.")
    print(f"Loaded {len(X)} prefixes ({len(np.unique(source_ids))} distinct questions)")

    train_mask, test_mask = group_train_test_split(source_ids, test_frac=args.test_frac, seed=args.seed)
    y = (n_wrong_arr > 0).astype(int)

    if y[train_mask].sum() == 0 or (1 - y[train_mask]).sum() == 0:
        sys.exit("Train split has only one class — increase sample size / --test-frac.")

    pca = PCA(n_components=min(args.pca_dim, train_mask.sum() - 1, X.shape[1]))
    Z_train = pca.fit_transform(X[train_mask])
    Z_test = pca.transform(X[test_mask])

    lr = LogisticRegression(max_iter=2000)
    lr.fit(Z_train, y[train_mask])

    train_acc = lr.score(Z_train, y[train_mask])
    test_scores = lr.predict_proba(Z_test)[:, 1]
    print(f"Train accuracy (corrupted vs not): {train_acc:.4f}")
    print(f"Train n={train_mask.sum()}, test n={test_mask.sum()}")

    report_trend(K_arr[test_mask], n_wrong_arr[test_mask], test_scores, label="HELD-OUT")

    if not args.skip_position_check:
        X_pos, K_pos, j_pos, n_eff_pos, src_pos = load_all_prefix_positions(args.hidden_dir)
        pos_test_mask = np.isin(src_pos, source_ids[test_mask])
        print(f"\nHELD-OUT, ALL intermediate prefixes (not just final h_K): "
              f"{pos_test_mask.sum()} rows from held-out source_ids")
        Z_pos_test = pca.transform(X_pos[pos_test_mask])
        pos_test_scores = lr.predict_proba(Z_pos_test)[:, 1]
        report_position_trend(K_pos[pos_test_mask], j_pos[pos_test_mask], n_eff_pos[pos_test_mask],
                               pos_test_scores, label="HELD-OUT, same probe scored at every prefix position")


if __name__ == "__main__":
    main()
