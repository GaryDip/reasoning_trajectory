"""
Shared loading / plotting helpers for the preliminary analyses (PRELIM_ANALYSIS_PLAN.md).

All four experiments reuse the hidden states that were already extracted for gate training
(`hidden_states/pilot_multilayer/{train_pilot,dev}`), so none of them needs a GPU except
experiment 4, which scores *natural* retrieval errors and therefore needs a fresh extraction.

Layout of one .npz (one reasoning chain):
    hidden     (n_layer, n_prefix, 4096)   last-token state of each cumulative prefix
                                           NOTE the layer axis comes FIRST
    layers     (n_layer,)                  [7, 15, 23, 31]
    trace_pos  (n_prefix,)                 1-indexed: 1 = question only, p = after step p-1
    K, trace_type, source_id, wrong_hops, first_wrong_hop

Note on indexing: PRELIM_ANALYSIS_PLAN.md calls these pos 0..K; the stored `trace_pos` is
1..K+1. Everything below uses the stored convention and figures say "prefix index".

Gold chains store every prefix (1..K+1). Counterfactual chains are truncated at the injected
hop: a chain with wrong hop h stores 1..h+1, so the state that first contains the substituted
passage is the LAST one, at trace_pos = h+1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HS = ROOT / "hidden_states" / "pilot_multilayer"
LAYER = 15  # the production layer (WORK_SUMMARY 2.2)


def manifest(split: str) -> list[dict]:
    return [json.loads(l) for l in (HS / split / "manifest.jsonl").open() if l.strip()]


def layer_index(z) -> int:
    layers = [int(x) for x in z["layers"]]
    return layers.index(LAYER)


def load_states(row: dict, split: str, positions: list[int] | None = None):
    """Return (states, trace_pos) at LAYER for one chain; `positions` filters trace_pos."""
    z = np.load(HS / split / row["path"], allow_pickle=True)
    li = layer_index(z)
    h = z["hidden"][li, :, :].astype(np.float32)   # (n_prefix, 4096)
    tp = np.asarray(z["trace_pos"], dtype=int)
    if positions is not None:
        keep = np.isin(tp, positions)
        h, tp = h[keep], tp[keep]
    return h, tp


def wrong_hop(row: dict) -> int | None:
    wh = row.get("wrong_hops") or []
    if wh:
        return int(wh[0])
    v = row.get("first_wrong_hop")
    return int(v) if v not in (None, -1) else None


# ---------------------------------------------------------------- plotting
FIG_W_CM = 16.0
CM = 1 / 2.54


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    })
    return plt


def save(fig, name: str):
    """Write both .png (quick viewing) and .pdf (vector, for LaTeX \\includegraphics)."""
    base = Path(__file__).resolve().parent / "figures" / Path(name).stem
    for ext in (".png", ".pdf"):
        fig.savefig(base.with_suffix(ext))
    print(f"  figure -> {base}.png / .pdf")
