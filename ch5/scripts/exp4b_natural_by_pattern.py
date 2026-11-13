#!/usr/bin/env python3
"""
CH5_PLAN task 2 -- prefix contamination under NATURAL errors, grouped by error PATTERN.

Experiment 4 grouped chains by "contains an error at hop h", so a chain with errors at hops 1 and 2
counted in both groups and the groups could not be read separately. Here a chain belongs to exactly
one group, its full tuple of wrong hops, so "only hop 1 is wrong" never mixes with "hops 1 and 2 are
wrong".

Two scores per transition, so the conclusion does not rest on one definition:
  * the Mahalanobis z of Δ against the correct-transition reference (as in experiment 4), and
  * the production gate v3 score itself, which makes the result directly comparable with the
    constructed-error analysis of Chapter 5.

Each cell is reported as the difference against the control group (chains of the same K with no
wrong hop at all) on the same transition, with a 95% bootstrap CI on the difference. Cells with
fewer than 30 chains are marked do_not_interpret.

CPU only: the hidden states were extracted by experiment 4.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prelim_analysis"))
import common  # noqa: E402

assert common.ROOT == ROOT, (common.ROOT, ROOT)
sys.path.insert(0, str(ROOT / "retrieval"))
sys.path.insert(0, str(ROOT / "gate"))
from run_retrieval_exp_wavefront_gate_v3 import score_gate_v3  # noqa: E402
from lr_artifacts import load_pooled_artifacts  # noqa: E402

NAT = common.HS.parent / "exp4_natural"
MIN_N = 30


def load_natural():
    rows = [json.loads(l) for l in (NAT / "manifest.jsonl").open() if l.strip()]
    out = []
    for r in rows:
        z = np.load(NAT / r["path"], allow_pickle=True)
        h = (z["hidden"][[int(x) for x in z["layers"]].index(common.LAYER)]
             if "layers" in z.files else z["hidden"]).astype(np.float32)
        tp = np.asarray(z["trace_pos"], dtype=int)
        order = np.argsort(tp)
        out.append({"K": int(r["K"]),
                    "wrong": tuple(sorted(int(x) for x in (r.get("wrong_hops") or []))),
                    "h": h[order]})
    return out


def reference(ref_split: str, max_chains: int, pca_dim: int):
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    acc = defaultdict(list)
    n = 0
    for r in common.manifest(ref_split):
        if r["trace_type"] != "correct":
            continue
        h, tp = common.load_states(r, ref_split)
        o = np.argsort(tp)
        h, tp = h[o], tp[o]
        for i in range(len(tp) - 1):
            acc[(int(r["K"]), int(tp[i]))].append(h[i + 1] - h[i])
        n += 1
        if n >= max_chains:
            break
    model = {}
    for key, D in acc.items():
        D = np.stack(D)
        nc = min(pca_dim, D.shape[0] - 1, D.shape[1])
        pca = PCA(n_components=nc, random_state=0).fit(D.astype(np.float64))
        Z = pca.transform(D.astype(np.float64))
        lw = LedoitWolf().fit(Z)
        P = np.linalg.pinv(lw.covariance_)
        dref = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Z - lw.location_, P, Z - lw.location_), 0))
        model[key] = {"pca": pca, "m": lw.location_, "P": P,
                      "mu": float(dref.mean()), "sd": float(dref.std() + 1e-9)}
    print(f"  reference fitted on {n} gold chains, {len(model)} (K, transition) cells")
    return model


def diff_ci(a, b, n=2000, seed=0):
    """95% CI on mean(a) - mean(b) by independent bootstrap of both groups."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    da = rng.choice(a, (n, len(a)), replace=True).mean(1)
    db = rng.choice(b, (n, len(b)), replace=True).mean(1)
    d = da - db
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-split", default="train_pilot")
    ap.add_argument("--ref-chains", type=int, default=8000)
    ap.add_argument("--pca-dim", type=int, default=40)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "ch5/results/exp4b_natural_by_pattern.json")
    args = ap.parse_args()

    print("Fitting the correct-transition reference ...")
    model = reference(args.ref_split, args.ref_chains, args.pca_dim)
    arts, _meta = load_pooled_artifacts(ROOT / "gate/layer15_uniform/v3_artifacts")

    print("Scoring the natural-error chains ...")
    chains = load_natural()
    # (K, wrong-tuple, transition) -> list of (z, gate score)
    cell = defaultdict(lambda: {"z": [], "gate": []})
    for c in chains:
        K, wrong, h = c["K"], c["wrong"], c["h"]
        for t in range(1, len(h)):                     # transition t ingests hop t
            key = (K, t)
            if key not in model:
                continue
            mm = model[key]
            z = mm["pca"].transform((h[t] - h[t - 1]).reshape(1, -1).astype(np.float64))[0]
            dif = z - mm["m"]
            dist = float(np.sqrt(max(float(dif @ mm["P"] @ dif), 0.0)))
            g = float(score_gate_v3(h[t], h[t - 1], arts[t - 1])) if (t - 1) in arts else float("nan")
            cell[(K, wrong, t)]["z"].append((dist - mm["mu"]) / mm["sd"])
            cell[(K, wrong, t)]["gate"].append(g)

    targets = [
        ("correct hop after an error", 2, (1,), [2]),
        ("correct hop after an error", 3, (1,), [2, 3]),
        ("correct hop after an error", 3, (2,), [3]),
        ("the hop that is wrong", 2, (2,), [2]),
        ("the hop that is wrong", 3, (3,), [3]),
        ("hop before the error (selection effect)", 2, (2,), [1]),
        ("hop before the error (selection effect)", 3, (3,), [1, 2]),
    ]
    out = {"min_n_to_interpret": MIN_N, "cells": []}
    print(f"\n{'group':40s} {'K':>2s} {'pattern':9s} {'t':>2s} {'n':>5s} "
          f"{'Δz':>7s} {'95% CI':>18s} {'Δgate':>7s} {'95% CI':>18s}")
    for name, K, pat, ts in targets:
        for t in ts:
            tgt = cell.get((K, pat, t))
            ctl = cell.get((K, (), t))
            if not tgt or not ctl:
                continue
            n_t, n_c = len(tgt["z"]), len(ctl["z"])
            dz = float(np.mean(tgt["z"]) - np.mean(ctl["z"]))
            dg = float(np.nanmean(tgt["gate"]) - np.nanmean(ctl["gate"]))
            ciz = diff_ci(tgt["z"], ctl["z"])
            cig = diff_ci(tgt["gate"], ctl["gate"])
            flag = n_t < MIN_N
            out["cells"].append({
                "group": name, "K": K, "wrong_hops": list(pat), "transition": t,
                "n_target": n_t, "n_control": n_c,
                "delta_z": round(dz, 3), "delta_z_ci95": [round(ciz[0], 3), round(ciz[1], 3)],
                "delta_gate": round(dg, 4),
                "delta_gate_ci95": [round(cig[0], 4), round(cig[1], 4)],
                "do_not_interpret": flag})
            print(f"{name:40s} {K:>2d} {str(pat):9s} {t:>2d} {n_t:>5d} {dz:>+7.2f} "
                  f"[{ciz[0]:+.2f},{ciz[1]:+.2f}]".ljust(0) +
                  f" {dg:>+7.3f} [{cig[0]:+.3f},{cig[1]:+.3f}]" + ("   do-not-interpret" if flag else ""))
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
