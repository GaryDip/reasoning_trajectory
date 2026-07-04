#!/usr/bin/env python3
"""
3D transition visualization.

Left:  PCA on raw 4096-d delta -> PC1/2/3; cubic axes; gate boundary plane in PCA-3D.
Right: 3D t-SNE on gate PCA-64; same face-on camera.

Usage:
  python gate/viz/plot_transition_3d.py --js 0
  python gate/viz/plot_transition_3d.py --save-html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.linear_model import LinearRegression
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
    linear_plane_mesh_in_pca3,
    maybe_subsample_rows,
    pca3_raw_deltas,
    run_tsne_3d,
    sample_lr_hyperplane_grid,
    score_rows_pooled,
)
from fit_lr_gate_pooled import pooled_transition_label  # noqa: E402
from lr_artifacts import load_pooled_artifacts  # noqa: E402

COLOR_POS = "#e74c3c"
COLOR_NEG = "#3498db"
COLOR_PLANE = (0.15, 0.75, 0.35, 0.4)
VIEW_ELEV = 6.0
VIEW_AZIM = -90.0


def _embed_matrix(rows: list[dict], mode: str) -> np.ndarray:
    if mode == "raw_tsne":
        return np.stack([r["delta"] for r in rows], axis=0).astype(np.float64)
    return np.stack([r["z_pca"] for r in rows], axis=0).astype(np.float64)


def _rotation_matrix_from_vectors(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """Rotation matrix R with R @ from_vec = to_vec (both 3-vectors)."""
    a = np.asarray(from_vec, dtype=np.float64).reshape(3)
    b = np.asarray(to_vec, dtype=np.float64).reshape(3)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.eye(3)
    a, b = a / na, b / nb
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        perp = perp - a * float(np.dot(perp, a))
        perp /= max(float(np.linalg.norm(perp)), 1e-12)
        return np.eye(3) - 2.0 * np.outer(perp, perp)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _separation_axis(xyz: np.ndarray, y: np.ndarray) -> np.ndarray:
    if y.sum() == 0 or (~y).sum() == 0:
        return np.array([1.0, 0.0, 0.0])
    d = xyz[y].mean(axis=0) - xyz[~y].mean(axis=0)
    n = float(np.linalg.norm(d))
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0])
    return d / n


def _apply_rotation(xyz: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (R @ xyz.T).T


def _apply_rotation_grid(grid: np.ndarray, R: np.ndarray) -> np.ndarray:
    sh = grid.shape
    flat = grid.reshape(-1, 3)
    return _apply_rotation(flat, R).reshape(sh)


def _plot_scatter3d(ax, xyz, mask, color, label, size=10, alpha=0.7):
    if not mask.any():
        return
    ax.scatter(
        xyz[mask, 0],
        xyz[mask, 1],
        xyz[mask, 2],
        c=color,
        s=size,
        alpha=alpha,
        label=label,
        depthshade=True,
        edgecolors="none",
    )


def _set_cubic_view(ax, xyz: np.ndarray) -> None:
    """Equal-scale axes + camera along +x (separation axis after rotation)."""
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    if xyz.size == 0:
        return
    span = np.percentile(np.abs(xyz - np.median(xyz, axis=0)), 98, axis=0)
    span = np.maximum(span, 1e-6)
    r = float(np.max(span))
    c = np.median(xyz, axis=0)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _gate_plane_in_pca3(P: np.ndarray, scores: np.ndarray, tau: float, grid_n: int):
    """Linear approximation: logit(gate score) ~ P; mesh where score = tau."""
    p = np.clip(scores, 1e-4, 1.0 - 1e-4)
    logit = np.log(p / (1.0 - p))
    reg = LinearRegression().fit(P, logit)
    target = float(np.log(tau / (1.0 - tau)))
    coef = reg.coef_
    intercept = float(reg.intercept_) - target
    mesh = linear_plane_mesh_in_pca3(P, coef, intercept, grid_n=grid_n)
    normal = coef / max(float(np.linalg.norm(coef)), 1e-12)
    return mesh, normal


def plot_j_3d(
    j: int,
    rows: list[dict],
    artifact: dict,
    *,
    split: str,
    embed: str,
    out_dir: Path,
    max_points: int,
    grid_n: int,
    tsne_iter: int,
    seed: int,
    save_html: bool,
) -> dict:
    sub = [r for r in rows if r["j"] == j and r["has_lr"]]
    sub = maybe_subsample_rows(sub, max_points=max_points, random_state=seed)
    if len(sub) < 50:
        print(f"  j={j}: skip (n={len(sub)})")
        return {"j": j, "skipped": True, "n": len(sub)}

    lr = artifact["lr"]
    tau = float(artifact["threshold"])
    trans = pooled_transition_label(j)

    Z_pca = np.stack([r["z_pca"] for r in sub])
    y = np.array([r["y_gold"] for r in sub], dtype=bool)
    scores = np.array([r["lr_score"] for r in sub])
    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        auc = float("nan")

    z_mean = Z_pca.mean(axis=0)
    D_raw = np.stack([r["delta"] for r in sub], axis=0).astype(np.float64)
    P_raw, pca_raw = pca3_raw_deltas(D_raw, random_state=seed)
    evr = pca_raw.explained_variance_ratio_
    evr_str = f"var={evr[0]:.0%}/{evr[1]:.0%}/{evr[2]:.0%}" if len(evr) >= 3 else ""

    plane_mesh, plane_normal = _gate_plane_in_pca3(P_raw, scores, tau, grid_n)
    sep = _separation_axis(P_raw, y)
    if plane_normal is not None and float(np.linalg.norm(plane_normal)) > 1e-12:
        sep = plane_normal
    R = _rotation_matrix_from_vectors(sep, np.array([1.0, 0.0, 0.0]))
    P_left = _apply_rotation(P_raw, R)
    if plane_mesh is not None:
        XX, YY, ZZ = plane_mesh
        plane_pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
        plane_pts = _apply_rotation(plane_pts, R)
        XX, YY, ZZ = plane_pts[:, 0].reshape(XX.shape), plane_pts[:, 1].reshape(YY.shape), plane_pts[:, 2].reshape(ZZ.shape)
    else:
        XX = YY = ZZ = None

    fig = plt.figure(figsize=(18, 8))

    # ── Left: raw 4096-d PCA ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(121, projection="3d")
    _plot_scatter3d(ax1, P_left, y, COLOR_POS, f"intervene (n={y.sum()})")
    _plot_scatter3d(ax1, P_left, ~y, COLOR_NEG, f"no intervene (n={(~y).sum()})")
    if XX is not None:
        ax1.plot_surface(
            XX, YY, ZZ,
            color=COLOR_PLANE[:3],
            alpha=COLOR_PLANE[3],
            linewidth=0,
            antialiased=True,
            shade=False,
        )
    ax1.set_xlabel("sep (aligned)")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.set_title(
        f"{trans} (j={j}) — PCA on raw delta (4096-d)\n"
        f"{evr_str}  gate tau={tau:.3f}  AUC={auc:.3f}"
    )
    _set_cubic_view(ax1, P_left)
    ax1.legend(loc="upper left", fontsize=8)

    # ── Right: 3D t-SNE ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(122, projection="3d")
    emb_plane = None

    if embed == "pca_tsne":
        Z_plane, _, _ = sample_lr_hyperplane_grid(
            lr, z_mean, grid_n=grid_n, z_cloud=Z_pca,
        )
        Z_plane_flat = Z_plane.reshape(-1, Z_pca.shape[1])
        Z_joint = np.vstack([Z_pca, Z_plane_flat])
        n_real = len(Z_pca)
        emb3 = run_tsne_3d(Z_joint, random_state=seed, max_iter=tsne_iter)
        emb_real = emb3[:n_real]
        emb_plane_raw = emb3[n_real:].reshape(grid_n, grid_n, 3)
    else:
        X_embed = _embed_matrix(sub, embed)
        emb_real = run_tsne_3d(X_embed, random_state=seed, max_iter=tsne_iter)
        emb_plane_raw = None

    sep = _separation_axis(emb_real, y)
    R = _rotation_matrix_from_vectors(sep, np.array([1.0, 0.0, 0.0]))
    emb_real = _apply_rotation(emb_real, R)
    if emb_plane_raw is not None:
        emb_plane = _apply_rotation_grid(emb_plane_raw, R)

    _plot_scatter3d(ax2, emb_real, y, COLOR_POS, f"intervene (n={y.sum()})", size=12)
    _plot_scatter3d(ax2, emb_real, ~y, COLOR_NEG, f"no intervene (n={(~y).sum()})", size=12)

    if emb_plane is not None:
        ax2.plot_surface(
            emb_plane[:, :, 0],
            emb_plane[:, :, 1],
            emb_plane[:, :, 2],
            color=COLOR_PLANE[:3],
            alpha=0.45,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        tsne_note = "LR sheet (joint 3D t-SNE, view ~ separation axis)"
    else:
        band = np.abs(scores - tau) < 0.04
        if band.sum() >= 10:
            _plot_scatter3d(
                ax2, emb_real, band, COLOR_PLANE[:3],
                f"|score-tau|<0.04 (n={band.sum()})", size=16, alpha=0.95,
            )
        tsne_note = "raw delta t-SNE (view ~ separation axis)"

    embed_label = "PCA-64 -> 3D t-SNE" if embed == "pca_tsne" else "raw delta -> 3D t-SNE"
    ax2.set_xlabel("separation (aligned)")
    ax2.set_ylabel("t-SNE axis 2")
    ax2.set_zlabel("t-SNE axis 3")
    ax2.set_title(f"{trans} (j={j}) — {embed_label}\n{tsne_note}")
    _set_cubic_view(ax2, emb_real)
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Transition delta  split={split}  n={len(sub)}  "
        f"(red=intervene, blue=no intervene; cubic axes, view along separation)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    png_path = out_dir / f"j{j}_{trans.replace('->', '_')}_3d.png"
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {png_path}")

    html_path = None
    if save_html:
        html_path = _save_plotly_html(
            j, trans, P_left, emb_real, emb_plane, y, auc, tau, out_dir,
        )

    return {
        "j": j,
        "transition": trans,
        "n": len(sub),
        "auc": round(auc, 4),
        "embed": embed,
        "png": str(png_path),
        "html": str(html_path) if html_path else None,
        "skipped": False,
    }


def _save_plotly_html(
    j: int,
    trans: str,
    pca_left: np.ndarray,
    emb_real: np.ndarray,
    emb_plane: np.ndarray | None,
    y: np.ndarray,
    auc: float,
    tau: float,
    out_dir: Path,
) -> Path | None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed; skip --save-html")
        return None

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=(
            f"Raw delta PCA-3D (AUC={auc:.3f})",
            f"3D t-SNE (tau={tau:.3f})",
        ),
    )

    def _add_cloud(fig, xyz, mask, color, name, row, col):
        fig.add_trace(
            go.Scatter3d(
                x=xyz[mask, 0],
                y=xyz[mask, 1],
                z=xyz[mask, 2],
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.75),
                name=name,
            ),
            row=row,
            col=col,
        )

    _add_cloud(fig, pca_left, y, COLOR_POS, "intervene", 1, 1)
    _add_cloud(fig, pca_left, ~y, COLOR_NEG, "no intervene", 1, 1)
    _add_cloud(fig, emb_real, y, COLOR_POS, "intervene", 1, 2)
    _add_cloud(fig, emb_real, ~y, COLOR_NEG, "no intervene", 1, 2)

    if emb_plane is not None:
        fig.add_trace(
            go.Surface(
                x=emb_plane[:, :, 0],
                y=emb_plane[:, :, 1],
                z=emb_plane[:, :, 2],
                colorscale=[[0, "rgba(40,190,90,0.5)"], [1, "rgba(40,190,90,0.5)"]],
                showscale=False,
                name="LR sheet",
            ),
            row=1,
            col=2,
        )

    camera = dict(eye=dict(x=2.2, y=0.0, z=0.15))
    fig.update_layout(title=f"j={j} {trans}", height=720, legend=dict(x=0, y=1))
    fig.update_scenes(aspectmode="data", camera=camera)
    html_path = out_dir / f"j{j}_{trans.replace('->', '_')}_3d.html"
    fig.write_html(str(html_path))
    print(f"  saved {html_path}")
    return html_path


def main() -> None:
    ap = argparse.ArgumentParser(description="3D transition t-SNE + LR boundary (face-on view)")
    ap.add_argument("--hs-root", type=Path, default=HS_ROOT)
    ap.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_POOLED)
    ap.add_argument("--split", default="dev", choices=("train", "dev"))
    ap.add_argument("--js", type=int, nargs="*", default=None)
    ap.add_argument(
        "--embed",
        default="pca_tsne",
        choices=("pca_tsne", "raw_tsne"),
    )
    ap.add_argument("--max-points", type=int, default=2000)
    ap.add_argument("--grid-n", type=int, default=35)
    ap.add_argument("--tsne-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "figures_3d")
    ap.add_argument("--save-html", action="store_true")
    args = ap.parse_args()

    models, meta = load_pooled_artifacts(args.artifacts_dir)
    js = args.js if args.js else sorted(models.keys())

    print(f"Loading {args.split} deltas ...")
    rows = collect_delta_rows_pooled(args.hs_root, args.split)
    scored = score_rows_pooled(rows, models)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for j in js:
        if j not in models:
            continue
        print(f"j={j} {pooled_transition_label(j)} ...")
        summaries.append(
            plot_j_3d(
                j,
                scored,
                models[j],
                split=args.split,
                embed=args.embed,
                out_dir=args.out_dir,
                max_points=args.max_points,
                grid_n=args.grid_n,
                tsne_iter=args.tsne_iter,
                seed=args.seed,
                save_html=args.save_html,
            )
        )

    summary_path = args.out_dir / f"transition_3d_summary_{args.split}.json"
    summary_path.write_text(
        json.dumps(
            {"split": args.split, "embed": args.embed, "panels": summaries, "meta": meta},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
