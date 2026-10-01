#!/usr/bin/env python3
"""How does the PRODUCTION gate (gate v3, PCA+LR on Delta_j) score a trace as the number of
non-gold evidences in it grows? Runs on the dev (validation) multi-error traces built by
build_multi_error_traces.py (n_wrong = 0..K variants of the same question) plus the layer-15/23
hidden states extract_full_hidden_states_multilayer.py writes for them.

Scoring mirrors production exactly: transition j uses layer meta["per_j_layers"][j], features are
concat(PCA(h_{j+1}), PCA(h_{j+1} - h_j)) and the score is the LR's P(abnormal) -- the same
score run_retrieval_exp_wavefront_gate_v3_rawprefix.py subtracts from the cosine score at rerank.

Three views, all on hops the gate actually has a model for (j <= 3, i.e. hops 1..4):
  1. hop level, by how many corrupted hops are in the prefix up to and including this hop
  2. hop level, by position relative to the FIRST corrupted hop (does an error leave a trace in
     the score of later, clean hops? -- production data could never answer this, see README)
  3. trace level, mean/max over the trace's hops vs n_wrong, plus Spearman

Usage:
  python score_gate_v3_vs_nwrong.py \
      --hidden-dir hidden_states_multilayer/dev --traces data/musique/dev_multi_error.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "retrieval"))

from run_retrieval_exp_wavefront_gate_v3 import (  # noqa: E402
    load_gate_v3_artifacts,
    score_gate_v3,
)


def score_parts(h_after: np.ndarray, h_prev: np.ndarray, art: dict) -> tuple[float, float, float]:
    """gate v3 fuses two halves: concat(PCA(h_after), PCA(Delta)). The LR logit is therefore
    exactly intercept + w_h . PCA(h_after) + w_delta . PCA(Delta), so each half's contribution to
    a hop's score can be read off separately -- which half reacts to a contaminated prefix?
    Returns (contribution of the h half, contribution of the Delta half, intercept)."""
    delta = (h_after - h_prev).astype(np.float64).reshape(1, -1)
    z_h = art["pca_h"].transform(h_after.astype(np.float64).reshape(1, -1))[0]
    z_d = art["pca_delta"].transform(delta)[0]
    w = art["lr"].coef_[0]
    n_h = z_h.shape[0]
    return float(w[:n_h] @ z_h), float(w[n_h:] @ z_d), float(art["lr"].intercept_[0])


def mean(v: list[float]) -> float:
    return float(np.mean(v)) if v else float("nan")


def fmt_cell(v: list[float], thr: float | None = None) -> str:
    if not v:
        return f"{'--':>18}"
    extra = f"/{np.mean([x > thr for x in v]):.2f}" if thr is not None else ""
    return f"{np.mean(v):.3f}±{np.std(v):.3f} (n={len(v)}){extra}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden-dir", type=Path, required=True,
                     help="multilayer hidden-state dir (must contain layers 15 and 23)")
    ap.add_argument("--traces", type=Path, default=HERE / "data/musique/dev_multi_error.jsonl")
    ap.add_argument("--artifacts-dir", type=Path,
                     default=PROJECT_ROOT / "gate/gate_v3/artifacts_pooled_v3")
    ap.add_argument("--limit", type=int, default=0, help="only score this many traces (smoke test)")
    ap.add_argument("--out", type=Path, default=HERE / "results/gate_v3_vs_nwrong_dev.json")
    args = ap.parse_args()

    meta = json.loads((args.artifacts_dir / "meta.json").read_text(encoding="utf-8"))
    per_j_layer = {int(j): int(l) for j, l in meta["per_j_layers"].items()}
    thr_by_j = {int(m["j"]): float(m["threshold"]) for m in meta["models"]}
    arts = load_gate_v3_artifacts(args.artifacts_dir)
    print(f"gate v3: per-j layers {per_j_layer}, thresholds "
          + ", ".join(f"j{j}={t:.3f}" for j, t in sorted(thr_by_j.items())))

    run_meta_path = args.hidden_dir / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8")) if run_meta_path.exists() else {}
    dir_layers = run_meta.get("layers") or ([run_meta["layer_0based"]] if "layer_0based" in run_meta else [])
    need = sorted(set(per_j_layer.values()))
    if dir_layers and not set(need) <= set(int(x) for x in dir_layers):
        sys.exit(f"{args.hidden_dir} has layers {dir_layers}, but gate v3 needs {need}. "
                 f"Extract them first with extract_full_hidden_states_multilayer.py --layers "
                 f"{','.join(str(l) for l in need)}")

    rows = {json.loads(l)["id"]: json.loads(l) for l in args.traces.open(encoding="utf-8") if l.strip()}
    manifest = [json.loads(l) for l in (args.hidden_dir / "manifest.jsonl").open(encoding="utf-8") if l.strip()]
    if args.limit:
        manifest = manifest[: args.limit]
    print(f"{len(manifest)} traces with hidden states, {len(rows)} trace rows")

    # hop records: (n_err_in_prefix, this_hop_wrong, offset_from_first_error, score, j, K, n_wrong, trace_id)
    hops: list[dict] = []
    per_trace: dict[str, list[float]] = defaultdict(list)
    trace_meta: dict[str, tuple[int, int]] = {}
    missing_layer = skipped = 0

    for m in manifest:
        row = rows.get(m["id"])
        if row is None:
            skipped += 1
            continue
        z = np.load(args.hidden_dir / m["path"], allow_pickle=True)
        layers = [int(x) for x in z["layers"]] if "layers" in z.files else [int(x) for x in dir_layers]
        hidden = z["hidden"]                      # (n_layers, n_prefix, hidden_dim)
        wrong = set(int(h) for h in (row.get("wrong_hops") or []))
        K, n_wrong = int(row["K"]), int(row["n_wrong"])
        first_err = min(wrong) if wrong else None
        trace_meta[m["id"]] = (K, n_wrong)
        for j in range(min(K, 4)):                # gate has models for j = 0..3 only
            layer = per_j_layer[j]
            if layer not in layers:
                missing_layer += 1
                continue
            li = layers.index(layer)
            if hidden.shape[1] < j + 2:           # need h_j and h_{j+1}
                continue
            s = score_gate_v3(hidden[li][j + 1], hidden[li][j], arts[j])
            c_h, c_d, c_b = score_parts(hidden[li][j + 1], hidden[li][j], arts[j])
            hop = j + 1                           # 1-based hop this transition commits
            hops.append({
                "score": s, "logit_h": c_h, "logit_delta": c_d, "logit_bias": c_b,
                "j": j, "hop": hop, "K": K, "n_wrong": n_wrong,
                "this_hop_wrong": hop in wrong,
                "n_err_in_prefix": sum(1 for w in wrong if w <= hop),
                "offset_from_first_error": None if first_err is None else hop - first_err,
            })
            per_trace[m["id"]].append(s)

    print(f"scored {len(hops)} hops from {len(per_trace)} traces "
          f"({skipped} traces missing a row, {missing_layer} hops missing their layer)\n")

    report: dict = {"n_hops": len(hops), "n_traces": len(per_trace)}

    # views 1 and 2 are reported pooled and per K (2/3/4-hop questions) -- a 4-hop question's
    # later hops sit on a longer prefix, so the same "1 error in the prefix" is not the same
    # situation across K.
    def subsets():
        yield "全部", (lambda h: True)
        for K in (2, 3, 4):
            yield f"K={K}", (lambda h, K=K: h["K"] == K)

    def view1_for(sel) -> dict:
        out = {}
        for n_err in range(0, 5):
            cells = {}
            for name, want in (("wrong", True), ("clean", False)):
                sub = [h for h in hops if sel(h) and h["n_err_in_prefix"] == n_err
                       and h["this_hop_wrong"] == want]
                v = [h["score"] for h in sub]
                thrs = [thr_by_j[h["j"]] for h in sub]
                cells[name] = {
                    "mean": mean(v), "n": len(v),
                    "above_threshold": float(np.mean([x > t for x, t in zip(v, thrs)])) if v else None,
                }
                cells[name + "_fmt"] = fmt_cell(v, float(np.mean(thrs)) if thrs else None)
            out[n_err] = cells
        return out

    def view2_for(sel) -> dict:
        out = {}
        for off in (-3, -2, -1, 0, 1, 2, 3):
            v = [h["score"] for h in hops if sel(h) and h["offset_from_first_error"] == off]
            if not v:
                continue
            v_clean = [h["score"] for h in hops if sel(h) and h["offset_from_first_error"] == off
                       and not h["this_hop_wrong"]]
            out[off] = {"mean": mean(v), "n": len(v),
                        "mean_clean_only": mean(v_clean), "n_clean": len(v_clean)}
        v0 = [h["score"] for h in hops if sel(h) and h["offset_from_first_error"] is None]
        out["no_error"] = {"mean": mean(v0), "n": len(v0)}
        return out

    OFF_LABELS = {-3: "错误前第 3 跳", -2: "错误前第 2 跳", -1: "错误前第 1 跳", 0: "错误发生的那一跳",
                  1: "错误后第 1 跳", 2: "错误后第 2 跳", 3: "错误后第 3 跳"}

    print("=== 1. 每跳分数 vs 该跳前缀里有几跳是错的（均值±标准差 (n) / 超阈值比例）===")
    report["by_n_err_in_prefix"] = {}
    for name, sel in subsets():
        v1 = view1_for(sel)
        report["by_n_err_in_prefix"][name] = v1
        print(f"\n-- {name} --")
        print(f"{'前缀中错误数':<14}{'当前跳是错的':>34}{'当前跳是对的':>34}")
        for n_err, cells in v1.items():
            if cells["wrong"]["n"] == 0 and cells["clean"]["n"] == 0:
                continue
            print(f"{n_err:<14}{cells['wrong_fmt']:>34}{cells['clean_fmt']:>34}")

    print("\n=== 2. 相对第一个错误跳的位置（只看 n_wrong>=1 的 trace）===")
    report["by_offset_from_first_error"] = {}
    for name, sel in subsets():
        v2 = view2_for(sel)
        report["by_offset_from_first_error"][name] = v2
        print(f"\n-- {name} --")
        for off, cell in v2.items():
            if off == "no_error":
                print(f"{'全 gold trace':<16} 全部 {cell['mean']:.3f} (n={cell['n']})")
                continue
            print(f"{OFF_LABELS[off]:<16} 全部 {cell['mean']:.3f} (n={cell['n']})"
                  f"    其中本身是对的 {cell['mean_clean_only']:.3f} (n={cell['n_clean']})")

    # view 2b: per-hop-position view -- is the score drifting simply because the prefix is longer?
    print("\n=== 2b. 每跳分数 vs 跳的位置（按 K 和这一跳本身对错分开）===")
    view2b = {}
    print(f"{'K':>3} {'hop':>4} {'本跳是错的':>26} {'本跳是对的(前缀全对)':>30} {'本跳是对的(前缀有错)':>30}")
    for K in (2, 3, 4):
        for hop in range(1, min(K, 4) + 1):
            base = [h for h in hops if h["K"] == K and h["hop"] == hop]
            if not base:
                continue
            cats = {
                "wrong": [h["score"] for h in base if h["this_hop_wrong"]],
                "clean_clean_prefix": [h["score"] for h in base if not h["this_hop_wrong"] and h["n_err_in_prefix"] == 0],
                "clean_dirty_prefix": [h["score"] for h in base if not h["this_hop_wrong"] and h["n_err_in_prefix"] > 0],
            }
            view2b[f"K{K}_hop{hop}"] = {k: {"mean": mean(v), "n": len(v)} for k, v in cats.items()}
            print(f"{K:>3} {hop:>4} " + " ".join(
                f"{(f'{mean(v):.3f} (n={len(v)})' if v else '--'):>26}" for v in cats.values()))
    report["by_hop_position"] = view2b

    # view 4: which half of the fused feature reacts? (logit = bias + w_h.PCA(h) + w_d.PCA(Delta))
    print("\n=== 4. fusion 两半各自的 logit 贡献（h 这一半 / Δ 这一半）===")
    print(f"{'类别':<26}{'n':>7}{'score':>8}{'logit(h)':>10}{'logit(Δ)':>10}{'截距':>8}")
    cats4 = [
        ("正确跳，前缀全对", lambda h: not h["this_hop_wrong"] and h["n_err_in_prefix"] == 0),
        ("正确跳，前缀有 1 错", lambda h: not h["this_hop_wrong"] and h["n_err_in_prefix"] == 1),
        ("正确跳，前缀有 2+ 错", lambda h: not h["this_hop_wrong"] and h["n_err_in_prefix"] >= 2),
        ("错误跳，前缀无其他错", lambda h: h["this_hop_wrong"] and h["n_err_in_prefix"] == 1),
        ("错误跳，前缀有其他错", lambda h: h["this_hop_wrong"] and h["n_err_in_prefix"] >= 2),
    ]
    view4 = {}
    for name, sel in cats4:
        sub = [h for h in hops if sel(h)]
        if not sub:
            continue
        view4[name] = {"n": len(sub), "score": mean([h["score"] for h in sub]),
                       "logit_h": mean([h["logit_h"] for h in sub]),
                       "logit_delta": mean([h["logit_delta"] for h in sub]),
                       "bias": mean([h["logit_bias"] for h in sub])}
        v = view4[name]
        print(f"{name:<26}{v['n']:>7}{v['score']:>8.3f}{v['logit_h']:>10.2f}{v['logit_delta']:>10.2f}{v['bias']:>8.2f}")
    report["fusion_halves"] = view4

    # view 3: trace level vs n_wrong
    print("\n=== 3. trace 级别：该 trace 所有跳的分数 vs n_wrong ===")
    print(f"{'K':>3} {'n_wrong':>8} {'n_traces':>9} {'mean(分数均值)':>16} {'mean(分数最大值)':>18}")
    view3 = {}
    means_all, maxs_all, nw_all = [], [], []
    for K in (2, 3, 4):
        for nw in range(0, K + 1):
            ids = [t for t, (k, n) in trace_meta.items() if k == K and n == nw and per_trace[t]]
            if not ids:
                continue
            mu = [float(np.mean(per_trace[t])) for t in ids]
            mx = [float(np.max(per_trace[t])) for t in ids]
            view3[f"K{K}_nwrong{nw}"] = {"n_traces": len(ids), "mean_of_mean": mean(mu), "mean_of_max": mean(mx)}
            means_all += mu
            maxs_all += mx
            nw_all += [nw] * len(ids)
            print(f"{K:>3} {nw:>8} {len(ids):>9} {mean(mu):>16.3f} {mean(mx):>18.3f}")
    from scipy.stats import spearmanr
    report["trace_level"] = view3
    if len(set(nw_all)) > 1 and len(nw_all) > 2:
        rho_mean = float(spearmanr(nw_all, means_all).statistic)
        rho_max = float(spearmanr(nw_all, maxs_all).statistic)
        report["spearman_nwrong_vs_trace_mean"] = rho_mean
        report["spearman_nwrong_vs_trace_max"] = rho_max
        print(f"\nSpearman(n_wrong, 分数均值) = {rho_mean:.3f}   Spearman(n_wrong, 分数最大值) = {rho_max:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
