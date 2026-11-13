#!/usr/bin/env python3
"""
Experiment 4 -- how far do NATURAL errors push a transition away from the "correct transition"
reference distribution?

Reference: for every (K, transition j), the Δ vectors of train gold chains, reduced with PCA and
summarised by a Ledoit-Wolf mean/covariance. Test: the chains the λ=0 pipeline actually produced
(exp4_build_traces.py), grouped by which hop went wrong. A chain with several wrong hops belongs
to several err_hop groups; chains with none form the control group.

Δ follows the production definition, Δ_j = h_j − h_{j−1} (gate/fit_lr_gate_pooled.py); the plan's
draft wrote "h_j − u_j", which is a typo.

Two deliberate differences from the original exploration:
  * No "→ Final Answer" column. In the original it carried the largest deviation of every row --
    including the control row (z ≈ 0.5-1.1) -- i.e. it mostly measured a mismatch between the
    reference chains and the pipeline chains, not an error signal. The stored prefixes stop
    before the final answer anyway, so every column here ingests a retrieved passage.
  * Raw Mahalanobis distances and the reference scale are reported next to the z-scores. The
    original heat map showed err_hop_1 deviating by only z=-0.05 at Q→E1, the very transition
    that ingests the wrong passage, while later columns deviated a lot. Without the reference
    spread per column one cannot tell whether that means "no signal" or "the reference for this
    column is simply very wide" -- the latter is likely, because Q→E1 is the largest transition.
    For contrast the trained gate is at its STRONGEST on that transition (j=0 train AUC 0.9707).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

import common

NAT_SPLIT = "exp4_natural"   # out-dir name used by the extraction step


def deltas_from_chain(h: np.ndarray, tp: np.ndarray):
    """[(transition j, Δ_j)] with j 1-indexed: transition j ingests step j."""
    order = np.argsort(tp)
    h, tp = h[order], tp[order]
    return [(int(tp[i]), h[i + 1] - h[i]) for i in range(len(tp) - 1)]


def collect_reference(split: str, max_chains: int):
    ref = defaultdict(list)
    n = 0
    for r in common.manifest(split):
        if r["trace_type"] != "correct":
            continue
        h, tp = common.load_states(r, split)
        for j, d in deltas_from_chain(h, tp):
            ref[(int(r["K"]), j)].append(d)
        n += 1
        if n >= max_chains:
            break
    print(f"  reference chains: {n}")
    return {k: np.stack(v) for k, v in ref.items()}


def collect_test(hs_dir):
    import json as _json
    rows = [_json.loads(l) for l in (hs_dir / "manifest.jsonl").open() if l.strip()]
    out = []
    for r in rows:
        z = np.load(hs_dir / r["path"], allow_pickle=True)
        # Two on-disk formats: the multi-layer pilot stores (n_layer, n_prefix, 4096) plus a
        # `layers` array; the single-layer extractor stores (n_prefix, 4096) and no `layers`.
        if "layers" in z.files:
            layers = [int(x) for x in z["layers"]]
            h = z["hidden"][layers.index(common.LAYER), :, :].astype(np.float32)
        else:
            h = z["hidden"].astype(np.float32)
        tp = np.asarray(z["trace_pos"], dtype=int)
        wrong = [int(x) for x in (r.get("wrong_hops") or [])]
        out.append((int(r["K"]), wrong, deltas_from_chain(h, tp)))
    print(f"  test chains: {len(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-split", default="train_pilot")
    ap.add_argument("--ref-chains", type=int, default=8000)
    ap.add_argument("--pca-dim", type=int, default=40)
    ap.add_argument("--hs-dir", type=common.Path,
                    default=common.HS.parent / NAT_SPLIT)
    args = ap.parse_args()

    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA

    print("Fitting the oracle reference ...")
    ref = collect_reference(args.ref_split, args.ref_chains)
    model = {}
    for (K, j), D in sorted(ref.items()):
        nc = min(args.pca_dim, D.shape[0] - 1, D.shape[1])
        pca = PCA(n_components=nc, random_state=0).fit(D.astype(np.float64))
        Z = pca.transform(D.astype(np.float64))
        lw = LedoitWolf().fit(Z)
        m = lw.location_
        P = np.linalg.pinv(lw.covariance_)
        d_ref = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Z - m, P, Z - m), 0))
        model[(K, j)] = {"pca": pca, "m": m, "P": P,
                         "mu": float(d_ref.mean()), "sd": float(d_ref.std() + 1e-9),
                         "n_ref": int(D.shape[0])}
        print(f"  K={K} j={j}: n={D.shape[0]:6d}  Mahalanobis mean {d_ref.mean():.2f} "
              f"sd {d_ref.std():.2f}")

    print("Scoring natural-error chains ...")
    test = collect_test(args.hs_dir)
    acc = defaultdict(list)      # (K, group, j) -> [z]
    raw = defaultdict(list)
    for K, wrong, ds in test:
        groups = ["control"] if not wrong else [f"err_hop_{h}" for h in wrong]
        for j, d in ds:
            key = (K, j)
            if key not in model:
                continue
            mm = model[key]
            z = mm["pca"].transform(d.reshape(1, -1).astype(np.float64))[0]
            diff = z - mm["m"]
            dist = float(np.sqrt(max(float(diff @ mm["P"] @ diff), 0.0)))
            for g in groups:
                acc[(K, g, j)].append((dist - mm["mu"]) / mm["sd"])
                raw[(K, g, j)].append(dist)

    res = {"layer": common.LAYER, "ref_split": args.ref_split, "pca_dim": args.pca_dim,
           "reference_scale": {f"K{K}_j{j}": {"mahalanobis_mean": round(v["mu"], 3),
                                              "mahalanobis_sd": round(v["sd"], 3),
                                              "n_ref": v["n_ref"]}
                               for (K, j), v in sorted(model.items())},
           "cells": {}}
    plt = common.setup_mpl()
    Ks = sorted({K for K, _, _ in acc})
    fig, axes = plt.subplots(1, len(Ks), constrained_layout=True,
                             figsize=(common.FIG_W_CM * common.CM, 5.0 * common.CM))
    for ax, K in zip(np.atleast_1d(axes), Ks):
        groups = ["control"] + [f"err_hop_{h}" for h in range(1, K + 1)]
        js = sorted({j for (k, _, j) in acc if k == K})
        M = np.full((len(groups), len(js)), np.nan)
        for gi, g in enumerate(groups):
            for ji, j in enumerate(js):
                v = acc.get((K, g, j))
                if v:
                    M[gi, ji] = float(np.mean(v))
                    res["cells"][f"K{K}_{g}_j{j}"] = {
                        "mean_z": round(float(np.mean(v)), 3),
                        "mean_mahalanobis": round(float(np.mean(raw[(K, g, j)])), 3),
                        "n": len(v)}
        im = ax.imshow(M, cmap="RdYlBu_r", vmin=-1.5, vmax=1.5)
        ax.set_xticks(range(len(js)), [f"E{j-1}→E{j}" if j > 1 else "Q→E1" for j in js])
        ax.set_yticks(range(len(groups)), groups)
        ax.set_title(f"K = {K}", fontsize=8)
        ax.grid(False)
        for gi in range(len(groups)):
            for ji in range(len(js)):
                if not np.isnan(M[gi, ji]):
                    ax.text(ji, gi, f"{M[gi, ji]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=0.8,
                 label="mean z-score vs oracle")
    fig.suptitle("Deviation of natural-error transitions from the correct-transition reference "
                 f"(Mahalanobis, layer {common.LAYER})", fontsize=8)
    common.save(fig, "exp4_oracle_manifold.png")
    plt.close(fig)

    # ---- second figure: deviation RELATIVE TO THE CONTROL ROW of the same column ----
    # The absolute z-scores are offset by a reference/test mismatch: the reference is fitted on
    # chains built from the annotated (GT) decomposition, while the test chains use the BART
    # decomposition used at inference. That offset shows up as a control row that is not 0
    # (clearly so for K=3 and K=4). Subtracting the control row of the same column cancels it,
    # leaving "how much further from the correct-transition manifold does an ERROR push this
    # transition, over and above the pipeline-vs-reference gap".
    fig, axes = plt.subplots(1, len(Ks), constrained_layout=True,
                             figsize=(common.FIG_W_CM * common.CM, 5.0 * common.CM))
    for ax, K in zip(np.atleast_1d(axes), Ks):
        js = sorted({j for (k, _, j) in acc if k == K})
        ctrl = np.array([np.mean(acc[(K, "control", j)]) if acc.get((K, "control", j)) else np.nan
                         for j in js])
        groups = [f"err_hop_{h}" for h in range(1, K + 1)]
        M = np.full((len(groups), len(js)), np.nan)
        for gi, g in enumerate(groups):
            for ji, j in enumerate(js):
                v = acc.get((K, g, j))
                if v:
                    M[gi, ji] = float(np.mean(v)) - ctrl[ji]
                    res["cells"][f"K{K}_{g}_j{j}"]["z_minus_control"] = round(float(M[gi, ji]), 3)
        im = ax.imshow(M, cmap="RdYlBu_r", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(js)), [f"E{j-1}>E{j}" if j > 1 else "Q>E1" for j in js], fontsize=6)
        ax.set_yticks(range(len(groups)), groups)
        n_ctrl = len(acc.get((K, "control", js[0]), []))
        ax.set_title(f"K = {K}   (control n = {n_ctrl})", fontsize=8)
        ax.grid(False)
        for gi in range(len(groups)):
            for ji in range(len(js)):
                if not np.isnan(M[gi, ji]):
                    ax.text(ji, gi, f"{M[gi, ji]:+.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=0.8,
                 label="z-score minus the control row")
    fig.suptitle("Same deviation, measured against the error-free chains of the SAME pipeline "
                 "(removes the reference/test offset)", fontsize=8)
    common.save(fig, "exp4_oracle_manifold_vs_control.png")
    plt.close(fig)
    res["note_control_offset"] = (
        "Absolute z-scores carry an offset because the reference uses GT decompositions and the "
        "test chains use the BART decomposition from the inference pipeline; the control row is "
        "therefore not 0 (about -0.4 for K=3 and -0.9 for K=4). The control-subtracted figure is "
        "the one to read.")

    out = common.Path(__file__).resolve().parent / "results" / "exp4_oracle_manifold.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"  results -> {out}")


if __name__ == "__main__":
    main()
