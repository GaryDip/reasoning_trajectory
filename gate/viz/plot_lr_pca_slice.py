#!/usr/bin/env python3
"""
Standalone PC1 x PC2 + LR score field (top-right panel only).

Usage:
  python gate/viz/plot_lr_pca_slice.py
  python gate/viz/plot_lr_pca_slice.py --gt-mode dual
  python gate/viz/plot_lr_pca_slice.py --gt-mode pos_dots --js 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
GATE_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(GATE_ROOT) not in sys.path:
    sys.path.insert(0, str(GATE_ROOT))

from transition_viz_utils import (  # noqa: E402
    ARTIFACTS_POOLED,
    HS_ROOT,
    collect_delta_rows_pooled,
    compute_pc12_score_grid,
    draw_pc12_tau_contour,
    load_pooled_artifacts,
    scatter_pc12_gt_binary,
    scatter_pc12_with_gt,
    score_rows_pooled,
)
from fit_lr_gate_pooled import pooled_transition_label  # noqa: E402


def _draw_pc12_field(
    ax,
    XX: np.ndarray,
    YY: np.ndarray,
    Sgrid: np.ndarray,
    tau: float,
    *,
    fig=None,
    show_cbar: bool = False,
) -> None:
    cf = ax.contourf(XX, YY, Sgrid, levels=20, cmap="RdYlBu_r", alpha=0.45, vmin=0, vmax=1)
    if show_cbar and fig is not None:
        fig.colorbar(cf, ax=ax, label="LR score", fraction=0.046, pad=0.04)
    if draw_pc12_tau_contour(ax, XX, YY, Sgrid, tau):
        ax.plot([], [], "k-", lw=2, label=f"τ={tau:.3f}")


def _draw_score_panel(
    ax,
    lr,
    Z: np.ndarray,
    scores: np.ndarray,
    y: np.ndarray,
    sub: list[dict],
    tau: float,
    *,
    gt_mode: str,
    fig=None,
    show_cbar: bool = True,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    if grid is None:
        XX, YY, Sgrid = compute_pc12_score_grid(lr, Z)
    else:
        XX, YY, Sgrid = grid
    _draw_pc12_field(ax, XX, YY, Sgrid, tau, fig=fig, show_cbar=show_cbar)
    scatter_pc12_with_gt(
        ax,
        Z,
        scores,
        y,
        rows=sub,
        gt_mode=gt_mode,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        tau=tau,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def _draw_gt_panel(
    ax,
    Z: np.ndarray,
    y: np.ndarray,
    tau: float,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    XX, YY, Sgrid = grid
    if draw_pc12_tau_contour(ax, XX, YY, Sgrid, tau):
        ax.plot([], [], "k-", lw=2, label=f"τ={tau:.3f}")
    scatter_pc12_gt_binary(ax, Z, y)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def plot_pca_slice(
    j: int,
    rows: list[dict],
    artifact: dict,
    *,
    split: str,
    out_dir: Path,
    gt_mode: str = "none",
) -> dict:
    sub = [r for r in rows if r["j"] == j and r["has_lr"]]
    if len(sub) < 30:
        print(f"  j={j}: skip (n={len(sub)})")
        return {"j": j, "n": len(sub), "skipped": True}

    lr = artifact["lr"]
    tau = float(artifact["threshold"])
    trans = pooled_transition_label(j)
    Z = np.stack([r["z_pca"] for r in sub])
    scores = np.array([r["lr_score"] for r in sub])
    y = np.array([r["y_gold"] for r in sub], dtype=bool)
    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        auc = float("nan")

    title_base = f"{trans} (j={j})  split={split}\nAUC={auc:.3f}  tau={tau:.3f}  n={len(sub)}"

    if gt_mode == "dual" and Z.shape[1] >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        grid = compute_pc12_score_grid(lr, Z)
        _draw_score_panel(
            ax1, lr, Z, scores, y, sub, tau,
            gt_mode="none", fig=fig, show_cbar=True, grid=grid,
        )
        ax1.set_title("LR score")
        ax1.legend(fontsize=8, loc="best")

        _draw_gt_panel(ax2, Z, y, tau, grid)
        ax2.set_title("GT label")
        ax2.legend(fontsize=8, loc="best")
        fig.suptitle(title_base, fontsize=11, y=1.02)
        fig.tight_layout()
        out_path = out_dir / f"j{j}_{trans.replace('->', '_')}_pca_slice_dual.png"
    elif Z.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        _draw_score_panel(
            ax, lr, Z, scores, y, sub, tau, gt_mode=gt_mode, fig=fig, show_cbar=True,
        )
        ax.set_title(title_base)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        suffix = f"_{gt_mode}" if gt_mode != "none" else ""
        out_path = out_dir / f"j{j}_{trans.replace('->', '_')}_pca_slice{suffix}.png"
    else:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        ax.text(0.5, 0.5, "need >=2 PCA dims", ha="center", va="center", transform=ax.transAxes)
        out_path = out_dir / f"j{j}_{trans.replace('->', '_')}_pca_slice.png"

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")

    return {
        "j": j,
        "transition": trans,
        "n": len(sub),
        "auc": round(auc, 4),
        "threshold": tau,
        "gt_mode": gt_mode,
        "figure": str(out_path),
        "skipped": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="PC1 x PC2 LR score field only")
    ap.add_argument("--hs-root", type=Path, default=HS_ROOT)
    ap.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_POOLED)
    ap.add_argument("--split", default="dev", choices=("train", "dev"))
    ap.add_argument("--js", type=int, nargs="*", default=None)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "pca_slice")
    ap.add_argument(
        "--gt-mode",
        default="none",
        choices=("none", "pos_dots", "inset", "dual", "marker", "confusion", "both"),
        help=(
            "GT viz: none | pos_dots (orange on pos) | inset (corner hist) | "
            "dual (score+GT panels) | confusion (TP/FP/FN rings)"
        ),
    )
    args = ap.parse_args()

    models, meta = load_pooled_artifacts(args.artifacts_dir)
    js = args.js if args.js else sorted(models.keys())

    print(f"Loading {args.split} deltas from {args.hs_root}")
    rows = collect_delta_rows_pooled(args.hs_root, args.split)
    scored = score_rows_pooled(rows, models)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for j in js:
        if j not in models:
            continue
        print(f"j={j} {pooled_transition_label(j)} ...")
        summaries.append(
            plot_pca_slice(
                j, scored, models[j], split=args.split, out_dir=args.out_dir, gt_mode=args.gt_mode,
            )
        )

    summary_path = args.out_dir / f"pca_slice_summary_{args.split}.json"
    summary_path.write_text(
        json.dumps({"split": args.split, "panels": summaries, "meta": meta}, indent=2),
        encoding="utf-8",
    )
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
