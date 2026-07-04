#!/usr/bin/env python3
"""
Visualize pooled LR decision planes after PCA on transition deltas.

For each semantic transition j, produces:
  1. LR-normal plane  — 2D coords (w direction, orthogonal); boundary is a vertical line
  2. PC1–PC2 slice    — scatter + linear decision boundary + score contours
  3. Score histogram  — pos vs neg + threshold τ
  4. t-SNE (optional) — in LR-PCA space, colored by score

Usage:
  python gate/viz/plot_lr_planes.py
  python gate/viz/plot_lr_planes.py --split dev --max-tsne 2000
  python gate/viz/plot_lr_planes.py --js 0 1 --out-dir gate/viz/results/figures
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
    CONF_COLORS,
    HS_ROOT,
    collect_delta_rows_pooled,
    confusion_bucket,
    draw_pc12_tau_contour,
    load_pooled_artifacts,
    lr_boundary_x,
    maybe_subsample_rows,
    project_to_lr_plane,
    run_tsne,
    scatter_pc12_with_gt,
    score_rows_pooled,
)
from fit_lr_gate_pooled import pooled_transition_label  # noqa: E402

K_COLORS = {2: "#1f77b4", 3: "#ff7f0e", 4: "#2ca02c"}


def plot_j_panel(
    j: int,
    rows: list[dict],
    artifact: dict,
    *,
    split: str,
    out_dir: Path,
    tsne_max: int,
    random_state: int,
) -> dict:
    sub = [r for r in rows if r["j"] == j and r["has_lr"]]
    if len(sub) < 30:
        print(f"  j={j}: skip (n={len(sub)})")
        return {"j": j, "n": len(sub), "skipped": True}

    pca = artifact["pca"]
    lr = artifact["lr"]
    tau = float(artifact["threshold"])
    Z = np.stack([r["z_pca"] for r in sub])
    scores = np.array([r["lr_score"] for r in sub])
    y = np.array([r["y_gold"] for r in sub], dtype=bool)
    Ks = np.array([r["K"] for r in sub], dtype=int)

    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        auc = float("nan")

    z_mean = Z.mean(axis=0)
    u_n, u_t = artifact.get("_plane_axes") or (None, None)
    if u_n is None:
        from transition_viz_utils import lr_plane_axes

        u_n, u_t = lr_plane_axes(lr, z_mean)
    plane_xy = project_to_lr_plane(Z, u_n, u_t)
    x_boundary = lr_boundary_x(lr, u_n)

    trans = pooled_transition_label(j)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    # ── A: LR-normal plane (true linear boundary = vertical line) ─────────────
    ax = fig.add_subplot(gs[0, 0])
    for K in sorted(set(Ks)):
        m = Ks == K
        ax.scatter(
            plane_xy[m, 0],
            plane_xy[m, 1],
            c=K_COLORS.get(K, "#888"),
            s=14,
            alpha=0.55,
            label=f"K={K}",
        )
    ax.axvline(x_boundary, color="crimson", ls="--", lw=1.5, label=f"logit=0 (τ→{tau:.2f})")
    # shade triggered side (score > 0.5 ≈ logit > 0)
    xlo, xhi = plane_xy[:, 0].min(), plane_xy[:, 0].max()
    pad = 0.08 * max(xhi - xlo, 1e-6)
    ax.axvspan(x_boundary, xhi + pad, alpha=0.06, color="red", label="intervene side")
    ax.set_xlabel("projection on LR normal (w/‖w‖)")
    ax.set_ylabel("orthogonal axis")
    ax.set_title(f"j={j} {trans} — LR decision plane\nAUC={auc:.3f}  τ={tau:.3f}  n={len(sub)}")
    ax.legend(fontsize=7, loc="best")

    # ── B: PC1–PC2 + boundary + contours ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    if Z.shape[1] >= 2:
        xlim = (float(Z[:, 0].min()), float(Z[:, 0].max()))
        ylim = (float(Z[:, 1].min()), float(Z[:, 1].max()))
        pad_x = 0.12 * max(xlim[1] - xlim[0], 1e-6)
        pad_y = 0.12 * max(ylim[1] - ylim[0], 1e-6)
        gxlim = (xlim[0] - pad_x, xlim[1] + pad_x)
        gylim = (ylim[0] - pad_y, ylim[1] + pad_y)
        w = np.asarray(lr.coef_, dtype=np.float64).reshape(-1)
        b = float(np.asarray(lr.intercept_).reshape(-1)[0])
        nc = min(Z.shape[1], w.size)
        xs_g = np.linspace(gxlim[0], gxlim[1], 70)
        ys_g = np.linspace(gylim[0], gylim[1], 70)
        XX, YY = np.meshgrid(xs_g, ys_g)
        Zgrid = np.zeros((XX.size, nc))
        Zgrid[:, 0] = XX.ravel()
        Zgrid[:, 1] = YY.ravel()
        Sgrid = lr.predict_proba(Zgrid)[..., 1].reshape(XX.shape)
        cf = ax.contourf(XX, YY, Sgrid, levels=20, cmap="RdYlBu_r", alpha=0.45, vmin=0, vmax=1)
        fig.colorbar(cf, ax=ax, label="LR score", fraction=0.046)
        if draw_pc12_tau_contour(ax, XX, YY, Sgrid, tau):
            ax.plot([], [], "k-", lw=2, label=f"τ={tau:.3f}")
        scatter_pc12_with_gt(
            ax, Z, scores, y, rows=sub, gt_mode="none", cmap="coolwarm", vmin=0, vmax=1,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA slice (PC1×PC2) + score field")
    ax.legend(fontsize=7, loc="best")

    # ── C: score histogram ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(scores[~y], bins=40, alpha=0.65, density=True, label=f"neg (n={(~y).sum()})")
    ax.hist(scores[y], bins=40, alpha=0.65, density=True, label=f"pos (n={y.sum()})")
    ax.axvline(tau, color="k", ls="--", lw=1.5, label=f"τ={tau:.3f}")
    ax.set_xlabel("LR score P(intervene)")
    ax.set_ylabel("density")
    ax.set_title("Score distribution")
    ax.legend(fontsize=8)

    # ── D: t-SNE in PCA space ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    tsne_rows = maybe_subsample_rows(sub, max_points=tsne_max, random_state=random_state)
    Z_ts = np.stack([r["z_pca"] for r in tsne_rows])
    if len(tsne_rows) >= 30:
        Z2 = run_tsne(Z_ts, random_state=random_state)
        ts_scores = np.array([r["lr_score"] for r in tsne_rows])
        ax.scatter(Z2[:, 0], Z2[:, 1], c=ts_scores, cmap="coolwarm", s=18, alpha=0.75, vmin=0, vmax=1)
        for bucket, marker, sz in [("TP", "o", 50), ("FP", "s", 40), ("FN", "X", 65)]:
            mask = np.array([confusion_bucket(r) == bucket for r in tsne_rows])
            if mask.any():
                ax.scatter(
                    Z2[mask, 0],
                    Z2[mask, 1],
                    s=sz,
                    facecolors="none",
                    edgecolors=CONF_COLORS[bucket],
                    linewidths=1.1,
                    label=f"{bucket} ({mask.sum()})",
                )
        ax.set_title(f"t-SNE in {Z_ts.shape[1]}-D PCA space")
    else:
        ax.text(0.5, 0.5, "too few points for t-SNE", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"Pooled LR gate — {trans} (j={j})  |  split={split}  "
        f"|  train_auc≈{artifact.get('train_auc', '?')}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out_path = out_dir / f"j{j}_{trans.replace('->', '_')}_lr_plane.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")

    return {
        "j": j,
        "transition": trans,
        "n": len(sub),
        "auc": round(auc, 4),
        "threshold": tau,
        "n_pos": int(y.sum()),
        "n_neg": int((~y).sum()),
        "figure": str(out_path),
        "skipped": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot pooled LR decision planes per transition j")
    ap.add_argument("--hs-root", type=Path, default=HS_ROOT)
    ap.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_POOLED)
    ap.add_argument("--split", default="dev", choices=("train", "dev"))
    ap.add_argument("--js", type=int, nargs="*", default=None, help="transition indices (default: all in meta)")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "figures")
    ap.add_argument("--max-tsne", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models, meta = load_pooled_artifacts(args.artifacts_dir)
    js = args.js if args.js else sorted(models.keys())

    print(f"Loading {args.split} deltas from {args.hs_root}")
    rows = collect_delta_rows_pooled(args.hs_root, args.split)
    scored = score_rows_pooled(rows, models)
    print(f"  {len(rows)} transitions, {sum(r['has_lr'] for r in scored)} scored")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for j in js:
        if j not in models:
            print(f"  j={j}: no model")
            continue
        art = models[j]
        from transition_viz_utils import lr_plane_axes

        u_n, u_t = lr_plane_axes(art["lr"], None)
        art = {**art, "_plane_axes": (u_n, u_t)}
        print(f"j={j} {pooled_transition_label(j)} ...")
        summaries.append(
            plot_j_panel(
                j,
                scored,
                art,
                split=args.split,
                out_dir=args.out_dir,
                tsne_max=args.max_tsne,
                random_state=args.seed,
            )
        )

    summary_path = args.out_dir / f"lr_plane_summary_{args.split}.json"
    summary_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "artifacts": str(args.artifacts_dir),
                "hs_root": str(args.hs_root),
                "meta": meta,
                "panels": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
