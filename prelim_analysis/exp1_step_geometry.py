#!/usr/bin/env python3
"""
Experiment 1 -- geometry of prefix states along a correct chain.

Original exploration (Obsidian screenshots): one pooled t-SNE of 12,000 last-layer states from
correct chains, coloured by prefix index 1..5; indices 1/2/3 separate cleanly, 4 and 5 overlap.

Two problems with reading that figure as "each step has its own distribution":
  (1) Prefix indices are NOT drawn from the same questions. Index 5 exists only for K=4 chains and
      index 4 only for K>=3, so the 4/5 overlap is partly "same questions, adjacent prefixes"
      rather than a property of step geometry. -> main figure is now faceted BY K, so within a
      panel every index comes from the same set of questions.
  (2) A t-SNE picture is not evidence. -> we also report k-NN purity of the prefix index in the
      ORIGINAL 4096-d space (t-SNE is only an illustration), on chains held out by question id.

Known confound, stated in the caption: prefix text grows monotonically with the index, and the
index is lexically visible in the text ("Step 1:", "Step 2:", ...). See exp2 for how far that
explains the structure.

Zero GPU: reuses hidden_states/pilot_multilayer/train_pilot (layer 15).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

import numpy as np

import common


def collect(split: str, per_K: int, seed: int):
    rng = random.Random(seed)
    by_K = defaultdict(list)
    for r in common.manifest(split):
        if r["trace_type"] == "correct":
            by_K[int(r["K"])].append(r)
    out = {}
    for K, rows in sorted(by_K.items()):
        rng.shuffle(rows)
        rows = rows[:per_K]
        X, pos, sid = [], [], []
        for r in rows:
            h, tp = common.load_states(r, split)
            X.append(h); pos.append(tp); sid += [r["source_id"]] * len(tp)
        out[K] = (np.concatenate(X), np.concatenate(pos), np.array(sid))
        print(f"  K={K}: {len(rows)} chains -> {out[K][0].shape[0]} prefixes")
    return out


def knn_purity(X, y, groups, k=10, seed=0):
    """Fraction of a point's k nearest neighbours (drawn from OTHER questions) sharing its label."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    ref_ids = set(uniq[: int(0.8 * len(uniq))])
    ref = np.array([g in ref_ids for g in groups])
    Z = PCA(n_components=50, random_state=0).fit_transform(X.astype(np.float64))
    nn = NearestNeighbors(n_neighbors=k).fit(Z[ref])
    _, idx = nn.kneighbors(Z[~ref])
    hit = (y[ref][idx] == y[~ref][:, None]).mean()
    # chance = P(a neighbour drawn at random from the reference set shares the query's label)
    chance = sum(np.mean(y[~ref] == c) * np.mean(y[ref] == c) for c in np.unique(y))
    return float(hit), float(chance)


def tsne(X, seed=0, perplexity=30):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    Z = PCA(n_components=50, random_state=0).fit_transform(X.astype(np.float64))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed,
                max_iter=1000).fit_transform(Z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train_pilot")
    ap.add_argument("--per-K", type=int, default=1000, help="chains sampled per K")
    ap.add_argument("--pooled-per-pos", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading gold chains from {args.split} (layer {common.LAYER}) ...")
    data = collect(args.split, args.per_K, args.seed)

    plt = common.setup_mpl()
    res = {"split": args.split, "layer": common.LAYER, "seed": args.seed,
           "tsne": {"perplexity": 30, "init": "pca", "pca_pre": 50, "max_iter": 1000}}

    # ---- main figure: one panel per K ----
    Ks = sorted(data)
    fig, axes = plt.subplots(1, len(Ks), figsize=(common.FIG_W_CM * common.CM, 5.6 * common.CM))
    cmap = plt.get_cmap("tab10")
    for ax, K in zip(np.atleast_1d(axes), Ks):
        X, pos, sid = data[K]
        E = tsne(X, seed=args.seed)
        for p in sorted(set(pos)):
            m = pos == p
            ax.scatter(E[m, 0], E[m, 1], s=1.5, alpha=0.5, linewidths=0,
                       color=cmap((p - 1) % 10), label=f"prefix {p}", rasterized=True)
        ax.set_title(f"K = {K} ({X.shape[0]:,} prefixes)")
        ax.set_xlabel("t-SNE 1"); ax.set_xticks([]); ax.set_yticks([])
        if K == Ks[0]:
            ax.set_ylabel("t-SNE 2")
        pur, chance = knn_purity(X, pos, sid)
        res.setdefault("knn_purity_by_K", {})[str(K)] = {
            "purity@10": round(pur, 4), "chance": round(chance, 4), "n": int(X.shape[0])}
        print(f"  K={K}: 10-NN purity {pur:.4f} (chance {chance:.4f})")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=3.5, color=cmap(p - 1),
                          label=f"prefix index {p}") for p in range(1, 6)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    common.save(fig, "exp1_step_geometry_by_K.png")
    plt.close(fig)

    # ---- companion: pooled across K, to match the original exploration ----
    Xs, ps = [], []
    rng = np.random.default_rng(args.seed)
    pool_X = np.concatenate([data[K][0] for K in Ks])
    pool_p = np.concatenate([data[K][1] for K in Ks])
    for p in sorted(set(pool_p)):
        idx = np.flatnonzero(pool_p == p)
        take = rng.choice(idx, size=min(args.pooled_per_pos, len(idx)), replace=False)
        Xs.append(pool_X[take]); ps.append(pool_p[take])
    X, pos = np.concatenate(Xs), np.concatenate(ps)
    E = tsne(X, seed=args.seed)
    fig, ax = plt.subplots(figsize=(common.FIG_W_CM * 0.6 * common.CM, 7.5 * common.CM))
    for p in sorted(set(pos)):
        m = pos == p
        ax.scatter(E[m, 0], E[m, 1], s=1.5, alpha=0.5, linewidths=0,
                   color=cmap((p - 1) % 10), label=f"prefix {p}", rasterized=True)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"All chain lengths pooled ({X.shape[0]:,} prefixes)", fontsize=8)
    ax.legend(markerscale=5, loc="upper left", framealpha=0.9, borderpad=0.3,
              handletextpad=0.3, labelspacing=0.25)
    common.save(fig, "exp1_step_geometry_pooled.png")
    plt.close(fig)
    res["pooled_counts"] = {str(p): int((pos == p).sum()) for p in sorted(set(pos))}

    out = common.Path(__file__).resolve().parent / "results" / "exp1_step_geometry.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"  results -> {out}")


if __name__ == "__main__":
    main()
