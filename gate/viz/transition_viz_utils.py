"""Utilities for pooled LR gate + transition delta visualization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

VIZ_ROOT = Path(__file__).resolve().parent
GATE_ROOT = VIZ_ROOT.parent
PROJECT_ROOT = GATE_ROOT.parent
HS_ROOT = PROJECT_ROOT / "hidden_states"
ARTIFACTS_POOLED = GATE_ROOT / "artifacts_pooled"

if str(GATE_ROOT) not in sys.path:
    sys.path.insert(0, str(GATE_ROOT))

from fit_lr_gate import collect_split, transition_label  # noqa: E402
from fit_lr_gate_pooled import pooled_transition_label  # noqa: E402
from hop_labels import should_intervene_at_j  # noqa: E402
from lr_artifacts import load_pooled_artifacts  # noqa: E402


def collect_delta_rows_pooled(
    hs_root: Path,
    split: str,
    *,
    j: int | None = None,
    K: int | None = None,
) -> list[dict]:
    """Flatten per-(K,j) groups into rows with gold label."""
    per_kj = collect_split(hs_root, split)
    rows: list[dict] = []
    for (K_tr, jj), group in sorted(per_kj.items()):
        if j is not None and jj != j:
            continue
        if K is not None and K_tr != K:
            continue
        if jj >= K_tr:
            continue
        for delta, label, meta in zip(group["X"], group["y"], group["meta"]):
            rows.append(
                {
                    "delta": np.asarray(delta, dtype=np.float32),
                    "K": int(K_tr),
                    "j": int(jj),
                    "split": split,
                    "example_id": meta["id"],
                    "trace_type": meta["trace_type"],
                    "transition": transition_label(jj, K_tr),
                    "pooled_transition": pooled_transition_label(jj),
                    "y_gold": bool(label),
                }
            )
    return rows


def score_rows_pooled(rows: list[dict], models: dict[int, dict]) -> list[dict]:
    out = []
    for r in rows:
        j = int(r["j"])
        y_gold = bool(r["y_gold"])
        if j not in models:
            out.append({**r, "lr_score": np.nan, "lr_triggered": False, "has_lr": False})
            continue
        art = models[j]
        pca = art["pca"]
        lr = art["lr"]
        tau = float(art["threshold"])
        z = pca.transform(np.asarray(r["delta"], dtype=np.float64).reshape(1, -1))[0]
        score = float(lr.predict_proba(z.reshape(1, -1))[0, 1])
        out.append(
            {
                **r,
                "z_pca": z,
                "lr_score": score,
                "lr_threshold": tau,
                "lr_triggered": bool(score > tau),
                "y_gold": y_gold,
                "has_lr": True,
            }
        )
    return out


def confusion_bucket(r: dict) -> str:
    y = bool(r.get("y_gold", False))
    pred = bool(r.get("lr_triggered", False))
    if y and pred:
        return "TP"
    if y and not pred:
        return "FN"
    if not y and pred:
        return "FP"
    return "TN"


CONF_COLORS = {"TP": "#2ca02c", "FP": "#ff7f0e", "TN": "#aec7e8", "FN": "#d62728"}


def draw_pc12_tau_contour(
    ax,
    XX: np.ndarray,
    YY: np.ndarray,
    Sgrid: np.ndarray,
    tau: float,
    *,
    color: str = "k",
    linewidth: float = 2.0,
) -> bool:
    """Draw score=τ contour on a PC1×PC2 LR score field (PC3…=0 slice)."""
    tau = float(np.clip(tau, 1e-4, 1.0 - 1e-4))
    if float(Sgrid.min()) > tau or float(Sgrid.max()) < tau:
        return False
    ax.contour(
        XX,
        YY,
        Sgrid,
        levels=[tau],
        colors=[color],
        linewidths=linewidth,
        linestyles="-",
    )
    return True


def compute_pc12_score_grid(
    lr,
    Z: np.ndarray,
    *,
    pad_frac: float = 0.12,
    grid_n: int = 70,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build PC1×PC2 score grid with PC3…=0 for contour / score-field plots."""
    xlim = (float(Z[:, 0].min()), float(Z[:, 0].max()))
    ylim = (float(Z[:, 1].min()), float(Z[:, 1].max()))
    pad_x = pad_frac * max(xlim[1] - xlim[0], 1e-6)
    pad_y = pad_frac * max(ylim[1] - ylim[0], 1e-6)
    gxlim = (xlim[0] - pad_x, xlim[1] + pad_x)
    gylim = (ylim[0] - pad_y, ylim[1] + pad_y)
    w = np.asarray(lr.coef_, dtype=np.float64).reshape(-1)
    nc = min(Z.shape[1], w.size)
    xs_g = np.linspace(gxlim[0], gxlim[1], grid_n)
    ys_g = np.linspace(gylim[0], gylim[1], grid_n)
    XX, YY = np.meshgrid(xs_g, ys_g)
    Zgrid = np.zeros((XX.size, nc))
    Zgrid[:, 0] = XX.ravel()
    Zgrid[:, 1] = YY.ravel()
    Sgrid = lr.predict_proba(Zgrid)[..., 1].reshape(XX.shape)
    return XX, YY, Sgrid


def scatter_pc12_gt_binary(ax, Z: np.ndarray, y_gold: np.ndarray) -> None:
    """GT-only panel: gray neg + red pos (same PC axes, no score colormap)."""
    xy = Z[:, :2]
    mask_neg = ~y_gold
    mask_pos = y_gold
    if mask_neg.any():
        ax.scatter(
            xy[mask_neg, 0],
            xy[mask_neg, 1],
            c="#bdbdbd",
            s=10,
            alpha=0.35,
            edgecolors="none",
            label=f"GT neg (n={int(mask_neg.sum())})",
        )
    if mask_pos.any():
        ax.scatter(
            xy[mask_pos, 0],
            xy[mask_pos, 1],
            c="#d62728",
            s=20,
            alpha=0.82,
            edgecolors="none",
            label=f"GT pos (n={int(mask_pos.sum())})",
        )


def add_pc12_score_inset(ax, scores: np.ndarray, y_gold: np.ndarray, tau: float) -> None:
    """Corner inset: pos/neg score histogram + τ."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    inset = inset_axes(ax, width="36%", height="36%", loc="upper right", borderpad=1.2)
    mask_neg = ~y_gold
    mask_pos = y_gold
    if mask_neg.any():
        inset.hist(scores[mask_neg], bins=22, alpha=0.55, density=True, color="#aec7e8")
    if mask_pos.any():
        inset.hist(scores[mask_pos], bins=22, alpha=0.75, density=True, color="#d62728")
    inset.axvline(tau, color="k", ls="-", lw=1.1)
    inset.set_xlim(0, 1)
    inset.set_yticks([])
    inset.tick_params(labelsize=6)
    inset.set_title("score | GT", fontsize=7, pad=2)


def scatter_pc12_with_gt(
    ax,
    Z: np.ndarray,
    scores: np.ndarray,
    y_gold: np.ndarray,
    *,
    rows: list[dict] | None = None,
    gt_mode: str = "marker",
    cmap: str = "coolwarm",
    vmin: float = 0,
    vmax: float = 1,
    tau: float | None = None,
) -> None:
    """
    PC1×PC2 scatter colored by LR score, with optional GT encoding.

    gt_mode:
      none      — score color only
      pos_dots  — score base + gold dots on GT pos only (clean single panel)
      inset     — score plot + corner pos/neg score histogram
      dual      — handled by caller (two-panel figure)
      marker    — legacy circle/triangle split
      confusion — score base + TP/FP/FN ring overlay (TN unmarked)
      both      — marker + confusion rings
    """
    xy = Z[:, :2]
    modes = {"none", "pos_dots", "inset", "marker", "confusion", "both"}
    if gt_mode not in modes:
        raise ValueError(f"gt_mode must be one of {sorted(modes)}")

    use_marker = gt_mode in ("marker", "both")
    use_conf = gt_mode in ("confusion", "both")

    if gt_mode == "none":
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=scores,
            cmap=cmap,
            s=14,
            alpha=0.85,
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
        )
        return

    if gt_mode == "pos_dots":
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=scores,
            cmap=cmap,
            s=12,
            alpha=0.55,
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
        )
        mask_pos = y_gold
        if mask_pos.any():
            ax.scatter(
                xy[mask_pos, 0],
                xy[mask_pos, 1],
                c="#e67e22",
                s=18,
                alpha=0.92,
                edgecolors="white",
                linewidths=0.25,
                label=f"GT pos (n={int(mask_pos.sum())})",
                zorder=3,
            )
        return

    if gt_mode == "inset":
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=scores,
            cmap=cmap,
            s=14,
            alpha=0.85,
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
        )
        if tau is None:
            tau = 0.5
        add_pc12_score_inset(ax, scores, y_gold, float(tau))
        return

    if use_marker:
        mask_neg = ~y_gold
        mask_pos = y_gold
        if mask_neg.any():
            ax.scatter(
                xy[mask_neg, 0],
                xy[mask_neg, 1],
                c=scores[mask_neg],
                cmap=cmap,
                marker="o",
                s=11,
                alpha=0.5,
                vmin=vmin,
                vmax=vmax,
                edgecolors="none",
                label=f"GT neg (n={int(mask_neg.sum())})",
            )
        if mask_pos.any():
            ax.scatter(
                xy[mask_pos, 0],
                xy[mask_pos, 1],
                c=scores[mask_pos],
                cmap=cmap,
                marker="^",
                s=40,
                alpha=0.95,
                vmin=vmin,
                vmax=vmax,
                edgecolors="k",
                linewidths=0.45,
                label=f"GT pos (n={int(mask_pos.sum())})",
            )
    else:
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=scores,
            cmap=cmap,
            s=12,
            alpha=0.7,
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
        )

    if use_conf and rows is not None:
        for bucket, marker, sz, lw in [
            ("TP", "o", 52, 1.15),
            ("FP", "s", 42, 1.1),
            ("FN", "X", 68, 1.25),
        ]:
            mask = np.array([confusion_bucket(r) == bucket for r in rows])
            if not mask.any():
                continue
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=sz,
                marker=marker,
                facecolors="none",
                edgecolors=CONF_COLORS[bucket],
                linewidths=lw,
                label=f"{bucket} ({int(mask.sum())})",
            )


def lr_plane_axes(lr, z_mean: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (u_normal, u_tangent) unit vectors in PCA space.

    u_normal aligns with LR weight w; decision boundary is u_normal · z = const.
    u_tangent is orthogonal (from PC1 with w removed).
    """
    w = np.asarray(lr.coef_, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(w))
    if norm < 1e-12:
        u_normal = np.zeros_like(w)
        u_normal[0] = 1.0
        norm = 1.0
    else:
        u_normal = w / norm

    ref = np.zeros_like(w)
    ref[0] = 1.0
    if z_mean is not None and z_mean.size == w.size:
        ref = np.asarray(z_mean, dtype=np.float64).reshape(-1)
    ref = ref - u_normal * float(np.dot(ref, u_normal))
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm < 1e-12:
        ref = np.zeros_like(w)
        ref[1 if w.size > 1 else 0] = 1.0
        ref = ref - u_normal * float(np.dot(ref, u_normal))
        ref_norm = float(np.linalg.norm(ref))
    u_tangent = ref / max(ref_norm, 1e-12)
    return u_normal, u_tangent


def project_to_lr_plane(z: np.ndarray, u_normal: np.ndarray, u_tangent: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        return np.array([float(np.dot(z, u_normal)), float(np.dot(z, u_tangent))])
    return np.stack([z @ u_normal, z @ u_tangent], axis=1)


def lr_boundary_x(lr, u_normal: np.ndarray) -> float:
    """x coordinate (along u_normal) where w·z + intercept = 0."""
    w = np.asarray(lr.coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(lr.intercept_).reshape(-1)[0])
    norm = float(np.linalg.norm(w))
    if norm < 1e-12:
        return 0.0
    return -intercept / norm


def score_grid_on_plane(
    pca,
    lr,
    u_normal: np.ndarray,
    u_tangent: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    n: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.stack(
        [x * u_normal + y * u_tangent for x, y in zip(xx.ravel(), yy.ravel())],
        axis=0,
    )
    scores = lr.predict_proba(zz)[..., 1].reshape(n, n)
    return xx, yy, scores


def maybe_subsample_rows(
    rows: list[dict],
    max_points: int = 0,
    random_state: int = 0,
) -> list[dict]:
    n = len(rows)
    if max_points <= 0 or n <= max_points:
        return rows
    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return [rows[i] for i in idx]


def _tsne_perplexity(n: int, perplexity: float | None) -> float:
    if perplexity is None:
        perplexity = min(30.0, max(5.0, (n - 1) / 3.0))
    return min(perplexity, n - 1)


def run_tsne(
    X: np.ndarray,
    *,
    n_components: int = 2,
    random_state: int = 0,
    perplexity: float | None = None,
    max_iter: int = 1000,
) -> np.ndarray:
    from sklearn.manifold import TSNE

    n = X.shape[0]
    if n < max(3, n_components + 1):
        raise ValueError(f"Need at least {n_components + 1} points for t-SNE, got {n}")
    tsne = TSNE(
        n_components=n_components,
        perplexity=_tsne_perplexity(n, perplexity),
        random_state=random_state,
        max_iter=max_iter,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X.astype(np.float64))


def run_tsne_3d(
    X: np.ndarray,
    *,
    random_state: int = 0,
    perplexity: float | None = None,
    max_iter: int = 1000,
) -> np.ndarray:
    return run_tsne(
        X,
        n_components=3,
        random_state=random_state,
        perplexity=perplexity,
        max_iter=max_iter,
    )


def lr_hyperplane_basis(
    lr,
    z_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthonormal basis (t1, t2, u_w) in PCA space; u_w = w/||w||.
    z_on_plane is the point on w·z+b=0 closest to z_mean.
    """
    w = np.asarray(lr.coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(lr.intercept_).reshape(-1)[0])
    z_mean = np.asarray(z_mean, dtype=np.float64).reshape(-1)
    w_norm = float(np.dot(w, w))
    if w_norm < 1e-12:
        u_w = np.zeros_like(w)
        u_w[0] = 1.0
        z_on_plane = z_mean.copy()
    else:
        u_w = w / np.sqrt(w_norm)
        z_on_plane = z_mean - w * (float(np.dot(w, z_mean)) + intercept) / w_norm

    tangents: list[np.ndarray] = []
    d = w.size
    for k in range(d):
        if len(tangents) >= 2:
            break
        e = np.zeros(d, dtype=np.float64)
        e[k] = 1.0
        v = e - u_w * float(np.dot(e, u_w))
        for t in tangents:
            v = v - t * float(np.dot(v, t))
        v_norm = float(np.linalg.norm(v))
        if v_norm > 1e-8:
            tangents.append(v / v_norm)
    while len(tangents) < 2:
        e = np.zeros(d, dtype=np.float64)
        e[len(tangents) + 1] = 1.0
        v = e - u_w * float(np.dot(e, u_w))
        for t in tangents:
            v = v - t * float(np.dot(v, t))
        tangents.append(v / max(float(np.linalg.norm(v)), 1e-8))

    return z_on_plane, tangents[0], tangents[1]


def sample_lr_hyperplane_grid(
    lr,
    z_mean: np.ndarray,
    *,
    grid_n: int = 40,
    span_std: float = 2.5,
    z_cloud: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a 2-D grid on the LR hyperplane w·z+b=0 in PCA space.

    Returns (Z_grid, aa, bb) with Z_grid shape (grid_n, grid_n, d).
    """
    z_on_plane, t1, t2 = lr_hyperplane_basis(lr, z_mean)
    if z_cloud is not None:
        z_cloud = np.asarray(z_cloud, dtype=np.float64)
        proj1 = z_cloud @ t1
        proj2 = z_cloud @ t2
        a_span = span_std * max(float(np.std(proj1)), 1e-6)
        b_span = span_std * max(float(np.std(proj2)), 1e-6)
    else:
        a_span = b_span = 3.0

    a = np.linspace(-a_span, a_span, grid_n)
    b = np.linspace(-b_span, b_span, grid_n)
    aa, bb = np.meshgrid(a, b)
    Z_grid = z_on_plane + aa[..., None] * t1 + bb[..., None] * t2
    return Z_grid, aa, bb


def project_to_lr_frame(
    Z: np.ndarray,
    lr,
    z_mean: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Coordinates (x, y, z) where x = projection on LR normal w/||w||,
    (y, z) span the decision hyperplane. Boundary is the plane x = x_boundary.
    """
    z_mean = np.asarray(z_mean, dtype=np.float64).reshape(-1)
    Z = np.asarray(Z, dtype=np.float64)
    _, t1, t2 = lr_hyperplane_basis(lr, z_mean)
    w = np.asarray(lr.coef_, dtype=np.float64).reshape(-1)
    u_w = w / max(float(np.linalg.norm(w)), 1e-12)
    x_boundary = lr_boundary_x(lr, u_w)
    frame = np.stack([Z @ u_w, Z @ t1, Z @ t2], axis=1)
    return frame, x_boundary


def lr_frame_plane_mesh(
    lr,
    z_mean: np.ndarray,
    z_cloud: np.ndarray,
    *,
    grid_n: int = 40,
    span_std: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Mesh for x = x_boundary in LR orthonormal frame (decision plane)."""
    z_mean = np.asarray(z_mean, dtype=np.float64)
    z_cloud = np.asarray(z_cloud, dtype=np.float64)
    _, t1, t2 = lr_hyperplane_basis(lr, z_mean)
    _, x_boundary = project_to_lr_frame(z_cloud[:1], lr, z_mean)
    proj_y = z_cloud @ t1
    proj_z = z_cloud @ t2
    y_span = span_std * max(float(np.std(proj_y)), 1e-6)
    z_span = span_std * max(float(np.std(proj_z)), 1e-6)
    yy = np.linspace(-y_span, y_span, grid_n)
    zz = np.linspace(-z_span, z_span, grid_n)
    YY, ZZ = np.meshgrid(yy, zz)
    XX = np.full_like(YY, x_boundary)
    return XX, YY, ZZ, x_boundary


def pca3_raw_deltas(D: np.ndarray, *, random_state: int = 0):
    """PCA on 4096-d delta vectors → 3D coordinates + fitted PCA object."""
    from sklearn.decomposition import PCA

    D = np.asarray(D, dtype=np.float64)
    n, d = D.shape
    nc = min(3, n - 1, d)
    pca = PCA(n_components=nc, random_state=random_state)
    P = pca.fit_transform(D)
    if nc < 3:
        pad = np.zeros((n, 3 - nc), dtype=np.float64)
        P = np.hstack([P, pad])
    return P, pca


def linear_plane_mesh_in_pca3(
    P: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    *,
    grid_n: int = 35,
    pad_frac: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Mesh for coef[0]*x + coef[1]*y + coef[2]*z + intercept = 0 in PCA-3D."""
    w = np.asarray(coef, dtype=np.float64).reshape(3)
    if abs(w[2]) < 1e-10:
        return None
    x, y = P[:, 0], P[:, 1]
    px = pad_frac * max(float(x.max() - x.min()), 1e-6)
    py = pad_frac * max(float(y.max() - y.min()), 1e-6)
    xx = np.linspace(float(x.min()) - px, float(x.max()) + px, grid_n)
    yy = np.linspace(float(y.min()) - py, float(y.max()) + py, grid_n)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = -(w[0] * XX + w[1] * YY + float(intercept)) / w[2]
    return XX, YY, ZZ


def load_metrics_summary(results_dir: Path | None = None) -> dict:
    results_dir = results_dir or (GATE_ROOT / "results_pooled")
    path = results_dir / "dev_lr_pooled_metrics.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
