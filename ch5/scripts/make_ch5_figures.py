#!/usr/bin/env python3
"""
Chapter 5 figures (CH5_PLAN.md task 3). Writes PDFs to ch5/figures/.

Same conventions as Chapter 4: setup_mpl() from prelim_analysis/common.py (8 pt text, 7 pt ticks
and legends), no suptitle, exact target width with a 1 mm margin, PDF.

    python ch5/scripts/make_ch5_figures.py          # both
    python ch5/scripts/make_ch5_figures.py 5.1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prelim_analysis"))
import common  # noqa: E402

OUT = ROOT / "ch5" / "figures"
CM = 1 / 2.54
HOPS = ROOT / "ch5/results/gate_l15_hops_dev.jsonl"
AGG = ROOT / "ch5/results/gate_l15_vs_nwrong_dev.json"   # same file as the probe's own copy
META = ROOT / "gate/layer15_uniform/v3_artifacts/meta.json"


def save(fig, name: str, width_cm: float, pad_cm: float = 0.10):
    """Exact width, guaranteed margin -- see pic/make_ch4_figures.py for why both are needed."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    target_in, pad_in = width_cm / 2.54, pad_cm / 2.54
    for _ in range(3):
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        if abs(tb.width + 2 * pad_in - target_in) < 0.004:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(*(np.array([w, h]) * target_in / (tb.width + 2 * pad_in)))
    fig.savefig(path, bbox_inches="tight", pad_inches=pad_in)
    import fitz
    r = fitz.open(path)[0].rect
    print(f"  -> {path}  ({r.width / 72 * 2.54:.2f} x {r.height / 72 * 2.54:.2f} cm)")


def boot_ci(xs, n=2000, seed=0):
    xs = np.asarray(xs, dtype=float)
    rng = np.random.default_rng(seed)
    m = rng.choice(xs, size=(n, len(xs)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---------------------------------------------------------------- 5.1
def fig_score_vs_error_position(plt):
    rows = [json.loads(l) for l in HOPS.open()]
    agg = json.loads(AGG.read_text())
    offsets = [-3, -2, -1, 0, 1, 2, 3]

    all_by, clean_by = {}, {}
    for r in rows:
        o = r["offset_from_first_error"]
        if o is None or o not in offsets:
            continue
        all_by.setdefault(o, []).append(r["score"])
        # the "correct hops" line is only meaningful after the error: before it every hop is
        # correct by definition (the two lines would coincide), and at offset 0 there are none
        if o > 0 and not r["this_hop_wrong"]:
            clean_by.setdefault(o, []).append(r["score"])

    fig, ax = plt.subplots(figsize=(9 * CM, 6 * CM), constrained_layout=True)

    gold = agg["by_offset_from_first_error"]["全部"]["no_error"]["mean"]
    ax.axhline(gold, color="0.45", ls="--", lw=0.8, zorder=1)
    ax.annotate(f"all-gold chains ({gold:.2f})", (offsets[-1], gold), textcoords="offset points",
                xytext=(-2, -3), fontsize=7, color="0.35", va="top", ha="right")
    # one decision threshold per hop position; show the average weighted by how many hops each
    # position contributes, since the x axis pools positions
    meta = json.loads(META.read_text())
    thr = {int(m["j"]): float(m["threshold"]) for m in meta["models"]}
    w = {}
    for r in rows:
        w[r["j"]] = w.get(r["j"], 0) + 1
    thr_avg = sum(thr[j] * c for j, c in w.items()) / sum(w.values())
    ax.axhline(thr_avg, color="0.7", ls=":", lw=0.8, zorder=1)
    ax.annotate(f"decision threshold ({thr_avg:.2f})", (offsets[0], thr_avg),
                textcoords="offset points", xytext=(2, 3), fontsize=7, color="0.45",
                va="bottom", ha="left")

    for label, data, color, ls, marker in [
        ("All hops", all_by, "tab:blue", "-", "o"),
        ("Correct hops only", clean_by, "tab:red", "--", "s"),
    ]:
        xs = [o for o in offsets if data.get(o)]
        ys = [float(np.mean(data[o])) for o in xs]
        lo = [y - boot_ci(data[o])[0] for y, o in zip(ys, xs)]
        hi = [boot_ci(data[o])[1] - y for y, o in zip(ys, xs)]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, ls=ls, marker=marker, ms=4, lw=1.2,
                    capsize=2, elinewidth=0.8, label=label, zorder=3)

    ax.set_xticks(offsets, [f"{o:+d}" if o else "0" for o in offsets])
    ax.set_xlabel("Hop position relative to the first error")
    ax.set_ylabel("Mean gate score")
    ax.set_ylim(0, 1)
    fig.legend(loc="outside lower center", ncol=2, frameon=False)
    save(fig, "disc_score_vs_error_position.pdf", 9)
    ns = {o: len(all_by.get(o, [])) for o in offsets}
    print(f"     n per offset: {ns}")
    plt.close(fig)


# ---------------------------------------------------------------- 5.2
def fig_logit_decomposition(plt):
    agg = json.loads(AGG.read_text())
    halves = agg["fusion_halves"]
    labels_en = {
        "正确跳，前缀全对": "Correct hop,\nclean prefix",
        "正确跳，前缀有 1 错": "Correct hop,\n1 error before",
        "正确跳，前缀有 2+ 错": "Correct hop,\n2+ errors before",
        "错误跳，前缀无其他错": "Wrong hop,\nno other error",
        "错误跳，前缀有其他错": "Wrong hop,\nother errors too",
    }
    keys = list(labels_en)
    h = [halves[k]["logit_h"] for k in keys]
    d = [halves[k]["logit_delta"] for k in keys]
    n = [halves[k]["n"] for k in keys]

    fig, ax = plt.subplots(figsize=(16 * CM, 5.5 * CM), constrained_layout=True)
    x = np.arange(len(keys))
    bw = 0.36
    ax.bar(x - bw / 2, h, bw, color="tab:blue", label=r"$w_h^\top\,\mathrm{PCA}(h)$", zorder=3)
    ax.bar(x + bw / 2, d, bw, color="tab:blue", alpha=0.4,
           label=r"$w_\Delta^\top\,\mathrm{PCA}(\Delta)$", zorder=3)
    ax.axhline(0, color="0.1", lw=0.9, zorder=4)
    for xi, v in zip(x - bw / 2, h):
        ax.text(xi, v + (0.12 if v >= 0 else -0.12), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7)
    for xi, v in zip(x + bw / 2, d):
        ax.text(xi, v + (0.12 if v >= 0 else -0.12), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7)
    ax.set_xticks(x, [f"{labels_en[k]}\n(n = {ni:,})" for k, ni in zip(keys, n)])
    ax.set_ylabel("Contribution to the logit")
    ax.set_ylim(-2.1, 4.0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "disc_logit_decomposition.pdf", 16)
    plt.close(fig)


FIGS = {"5.1": fig_score_vs_error_position, "5.2": fig_logit_decomposition}

if __name__ == "__main__":
    plt = common.setup_mpl()
    for k in (sys.argv[1:] or list(FIGS)):
        print(f"figure {k}:")
        FIGS[k](plt)
