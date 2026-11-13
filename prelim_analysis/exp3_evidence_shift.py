#!/usr/bin/env python3
"""
Experiment 3 -- does swapping in a wrong passage move the state, at a fixed position?

Original exploration: for each injected hop h, a shared t-SNE of ~100-150 prefix-aligned pairs
(same question, same prefix index), colour = prefix index, marker = correct / wrong evidence.

Problem with that figure: it is dominated by the prefix index. Points separate by colour, not by
marker, so it mostly re-shows experiment 1. The correct/wrong contrast is invisible because the
between-position variance is much larger than the within-position shift.

Fixes here:
  * one panel per (K, injected hop), i.e. position is held FIXED inside a panel, so the only
    factor left on the plot is correct vs substituted evidence;
  * two to three orders of magnitude more pairs than the original (405-14,376 per cell);
  * a quantitative companion, because a t-SNE is not evidence:
      - paired cosine distance between the gold state and its counterfactual twin, against the
        distance between two different questions' gold states at the same position (a SCALE
        reference, not a control: two different questions differ in everything);
      - separability, reported by citing the gate's own held-out AUC rather than fitting another
        probe on the same states (WORK_SUMMARY 1.4: h-only 0.9345 -- the h-only number is the
        right one here, since this experiment compares states, not transitions).

Alignment: a counterfactual chain with injected hop h stores prefixes 1..h+1, so the state that
first contains the substituted passage is at trace_pos = h+1; it is compared against the gold
chain of the same question at the same trace_pos.

Zero GPU.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import numpy as np

import common


def build_pairs(split: str, max_per_cell: int, seed: int):
    rng = random.Random(seed)
    gold_rows, cf_rows = {}, defaultdict(list)
    for r in common.manifest(split):
        if r["trace_type"] == "correct":
            gold_rows[r["source_id"]] = r
        else:
            h = common.wrong_hop(r)
            if h is not None:
                cf_rows[(int(r["K"]), h)].append(r)

    cells = {}
    for (K, h), rows in sorted(cf_rows.items()):
        rng.shuffle(rows)
        pos = h + 1  # prefix that first contains the substituted passage
        g_cache: dict[str, np.ndarray] = {}
        G, N, sid = [], [], []
        for r in rows:
            src = r["source_id"]
            if src not in gold_rows:
                continue
            if src not in g_cache:
                hg, tg = common.load_states(gold_rows[src], split, positions=[pos])
                if len(tg) == 0:
                    continue
                g_cache[src] = hg[0]
            hn, tn = common.load_states(r, split, positions=[pos])
            if len(tn) == 0:
                continue
            G.append(g_cache[src]); N.append(hn[0]); sid.append(src)
            if len(G) >= max_per_cell:
                break
        if G:
            cells[(K, h)] = (np.stack(G), np.stack(N), np.array(sid))
            print(f"  K={K} wrong hop={h} (prefix {pos}): {len(G)} pairs")
    return cells


def cosine(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return 1.0 - (a * b).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-per-cell", type=int, default=400)
    ap.add_argument("--tsne-per-cell", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Pairing gold / counterfactual states in {args.split} (layer {common.LAYER}) ...")
    cells = build_pairs(args.split, args.max_per_cell, args.seed)

    plt = common.setup_mpl()
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    res = {"split": args.split, "layer": common.LAYER, "cells": {}}
    rng = np.random.default_rng(args.seed)

    keys = sorted(cells)
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, constrained_layout=True,
                             figsize=(common.FIG_W_CM * common.CM, 4.0 * nrow * common.CM))
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, keys):
        K, h = key
        G, N, sid = cells[key]
        take = rng.permutation(len(G))[: args.tsne_per_cell]
        X = np.concatenate([G[take], N[take]])
        lab = np.array([0] * len(take) + [1] * len(take))
        Z = PCA(n_components=min(50, X.shape[0] - 1), random_state=0).fit_transform(X.astype(np.float64))
        E = TSNE(n_components=2, perplexity=30, init="pca", random_state=args.seed,
                 max_iter=1000).fit_transform(Z)
        ax.scatter(E[lab == 0, 0], E[lab == 0, 1], s=3, alpha=0.6, linewidths=0,
                   color="#2b6cb0", label="gold evidence", rasterized=True)
        # marker "x" is stroke-only: linewidths must be > 0 or nothing is drawn
        ax.scatter(E[lab == 1, 0], E[lab == 1, 1], s=4, alpha=0.6, linewidths=0.45,
                   color="#c53030", marker="x", label="substituted evidence", rasterized=True)
        ax.set_title(f"K={K}, hop {h} -> prefix {h+1}", fontsize=7, pad=2)
        ax.set_xticks([]); ax.set_yticks([])

        d_pair = cosine(G, N)
        perm = rng.permutation(len(G))
        same = perm == np.arange(len(G))
        perm[same] = (perm[same] + 1) % len(G)
        d_cross = cosine(G, G[perm])          # scale reference: two different questions
        d_cf_cross = cosine(G, N[perm])
        res["cells"][f"K{K}_h{h}"] = {
            "n_pairs": int(len(G)), "prefix_index": h + 1,
            "cos_dist_gold_vs_counterfactual_same_question": {
                "mean": round(float(d_pair.mean()), 4), "std": round(float(d_pair.std()), 4)},
            "cos_dist_gold_vs_gold_different_question": {
                "mean": round(float(d_cross.mean()), 4), "std": round(float(d_cross.std()), 4)},
            "cos_dist_gold_vs_counterfactual_different_question": {
                "mean": round(float(d_cf_cross.mean()), 4), "std": round(float(d_cf_cross.std()), 4)},
            "ratio_same_over_different": round(float(d_pair.mean() / d_cross.mean()), 4)}
        print(f"  K={K} h={h}: paired cos {d_pair.mean():.4f} | different-question cos "
              f"{d_cross.mean():.4f} | ratio {d_pair.mean()/d_cross.mean():.3f}")

    for ax in axes[len(keys):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=3, color="#2b6cb0", label="gold evidence"),
               plt.Line2D([], [], marker="x", ls="", ms=3, color="#c53030",
                          label="substituted evidence")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Gold vs substituted evidence at a fixed position "
                 f"(layer {common.LAYER}, {args.split})", fontsize=8)
    common.save(fig, "exp3_evidence_shift_by_cell.png")
    plt.close(fig)

    # ---- distance summary bar chart ----
    fig, ax = plt.subplots(figsize=(common.FIG_W_CM * common.CM, 5.0 * common.CM))
    labels = [f"K{K}\nhop {h}" for K, h in keys]
    xs = np.arange(len(keys))
    same_m = [res["cells"][f"K{K}_h{h}"]["cos_dist_gold_vs_counterfactual_same_question"]["mean"] for K, h in keys]
    diff_m = [res["cells"][f"K{K}_h{h}"]["cos_dist_gold_vs_gold_different_question"]["mean"] for K, h in keys]
    ax.bar(xs - 0.2, same_m, 0.4, label="gold vs substituted, same question", color="#c53030")
    ax.bar(xs + 0.2, diff_m, 0.4, label="gold vs gold, different question", color="#a0aec0")
    ax.set_xticks(xs, labels); ax.set_ylabel("cosine distance")
    ax.set_title("Size of the evidence-substitution shift, against a between-question scale "
                 "reference", fontsize=8)
    ax.legend(frameon=False)
    common.save(fig, "exp3_shift_distances.png")
    plt.close(fig)

    # ---- 1-D projection onto the paired shift direction (held-out halves) ----
    from sklearn.metrics import roc_auc_score
    fig, axes = plt.subplots(nrow, ncol, constrained_layout=True,
                             figsize=(common.FIG_W_CM * common.CM, 3.6 * nrow * common.CM))
    axes = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes, keys):
        K, h = key
        G, N, _ = cells[key]
        n = len(G); half = n // 2
        idx = rng.permutation(n)
        a, b = idx[:half], idx[half:]
        w = (N[a].mean(0) - G[a].mean(0))          # direction fitted on half A
        w /= np.linalg.norm(w) + 1e-9
        sg, sn = G[b] @ w, N[b] @ w                 # scored on held-out half B
        auc = roc_auc_score(np.r_[np.zeros(len(sg)), np.ones(len(sn))], np.r_[sg, sn])
        lo, hi = np.percentile(np.r_[sg, sn], [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        ax.hist(sg, bins=bins, alpha=0.6, color="#2b6cb0", label="gold")
        ax.hist(sn, bins=bins, alpha=0.6, color="#c53030", label="substituted")
        ax.set_title(f"K={K}, hop {h}   AUC = {auc:.3f}", fontsize=7, pad=2)
        ax.set_yticks([])
        res["cells"][f"K{K}_h{h}"]["shift_direction_auc_heldout"] = round(float(auc), 4)
        print(f"  K={K} h={h}: 1-D shift-direction AUC (held-out half) {auc:.4f}")
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#2b6cb0", alpha=0.6, label="gold evidence"),
               plt.Rectangle((0, 0), 1, 1, color="#c53030", alpha=0.6, label="substituted evidence")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    common.save(fig, "exp3_shift_projection.png")
    plt.close(fig)

    res["separability_note"] = (
        "Separability of gold vs substituted states at the same position is already measured by "
        "the gate's held-out classifier: h-only dev AUC 0.9345 (WORK_SUMMARY 1.4). No new probe "
        "is fitted here, to avoid reporting the same quantity twice.")
    out = common.Path(__file__).resolve().parent / "results" / "exp3_evidence_shift.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"  results -> {out}")


if __name__ == "__main__":
    main()
