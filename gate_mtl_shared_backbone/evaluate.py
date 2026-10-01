#!/usr/bin/env python3
"""Per-K (2/3/4-hop) validation metrics for all three heads. head1/head2 are binary
classification (AUC + accuracy@0.5); head3 is a [0,1] regression target (Pearson r + MSE
between the sigmoid output and the true final_answer_f1).

Reports two views per K: "all" (every trace: recipe1 gold-injected, recipe2 natural, recipe3
forced-wrong, recipe4 clean-prefix) and "natural" (recipe2 / is_natural_seed only) -- recipe1's
full-gold traces and recipe4's forced-gold prefix hops have evidence_label=1 by CONSTRUCTION,
not because the model figured anything out, so "all" can look better than the model's real
judgment ability on hops nobody told it the answer to in advance. "natural" is the harder, more
production-realistic number: real cosine retrieval, nothing forced either way."""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from packed_data import PackedCache, batch_from_indices


@torch.no_grad()
def _eval_indices(model, cache: PackedCache, idx: np.ndarray, batch_size: int) -> dict[str, float]:
    if len(idx) == 0:
        return {"n_traces": 0}

    ev_pred, ev_true = [], []
    ha_pred, ha_true = [], []
    f1_pred, f1_true = [], []
    for s in range(0, len(idx), batch_size):
        batch = batch_from_indices(cache, idx[s:s + batch_size])
        out = model(batch["x"])
        m = batch["mask"].bool()
        ev_pred.append(torch.sigmoid(out["evidence_logit"])[m].numpy())
        ev_true.append(batch["evidence_label"][m].numpy())
        ha_pred.append(torch.sigmoid(out["hop_answer_logit"])[m].numpy())
        ha_true.append(batch["hop_answer_label"][m].numpy())
        f1_pred.append(torch.sigmoid(out["final_f1_logit"])[m].numpy())
        f1_true.append(batch["final_f1_label"][m].numpy())

    ev_pred, ev_true = np.concatenate(ev_pred), np.concatenate(ev_true)
    ha_pred, ha_true = np.concatenate(ha_pred), np.concatenate(ha_true)
    f1_pred, f1_true = np.concatenate(f1_pred), np.concatenate(f1_true)

    metrics: dict[str, float] = {"n_traces": len(idx), "n_hops": len(ev_true)}
    for name, pred, true in [("evidence", ev_pred, ev_true), ("hop_answer", ha_pred, ha_true)]:
        # hop_answer's true label is continuous (hop_answer_f1), not strictly {0,1} like
        # evidence's is_correct -- roc_auc_score requires a binary ground truth, so threshold
        # at 0.5 for the purposes of this report table (training itself still uses the raw
        # continuous value as a soft BCE target, unaffected by this reporting-only binarization).
        true_bin = (true > 0.5).astype(np.float32)
        metrics[f"{name}_acc"] = float(((pred > 0.5) == (true_bin > 0.5)).mean())
        if len(np.unique(true_bin)) > 1:
            metrics[f"{name}_auc"] = float(roc_auc_score(true_bin, pred))
        else:
            metrics[f"{name}_auc"] = float("nan")  # only one class present in this bucket
    metrics["final_f1_mse"] = float(np.mean((f1_pred - f1_true) ** 2))
    if np.std(f1_true) > 0 and np.std(f1_pred) > 0:
        metrics["final_f1_pearson"] = float(pearsonr(f1_pred, f1_true)[0])
    else:
        metrics["final_f1_pearson"] = float("nan")
    return metrics


@torch.no_grad()
def evaluate_per_k(
    model, cache: PackedCache, val_case_ids: set[str], *, ks: tuple[int, ...] = (2, 3, 4), batch_size: int = 64,
    natural_only: bool = False,
) -> dict[int, dict[str, float]]:
    model.eval()
    results: dict[int, dict[str, float]] = {}
    for k in ks:
        idx = cache.indices_for_case_ids_and_k(val_case_ids, k, natural_only=natural_only)
        results[k] = _eval_indices(model, cache, idx, batch_size)
    model.train()
    return results


@torch.no_grad()
def evaluate_per_k_all_and_natural(
    model, cache: PackedCache, val_case_ids: set[str], *, ks: tuple[int, ...] = (2, 3, 4), batch_size: int = 64,
) -> dict[int, dict[str, dict[str, float]]]:
    """{k: {"all": metrics, "natural": metrics}}"""
    model.eval()
    results: dict[int, dict[str, dict[str, float]]] = {}
    for k in ks:
        idx_all = cache.indices_for_case_ids_and_k(val_case_ids, k, natural_only=False)
        idx_nat = cache.indices_for_case_ids_and_k(val_case_ids, k, natural_only=True)
        results[k] = {
            "all": _eval_indices(model, cache, idx_all, batch_size),
            "natural": _eval_indices(model, cache, idx_nat, batch_size),
        }
    model.train()
    return results


def _fmt_row(k, view, m) -> str:
    if m.get("n_traces", 0) == 0:
        return f"{k:>3} {view:>8} {'--':>6} (no val cases)"
    return (
        f"{k:>3} {view:>8} {m['n_traces']:>6} {m['evidence_auc']:>7.3f} {m['evidence_acc']:>7.3f} "
        f"{m['hop_answer_auc']:>7.3f} {m['hop_answer_acc']:>7.3f} "
        f"{m['final_f1_pearson']:>7.3f} {m['final_f1_mse']:>7.3f}"
    )


def format_per_k_table(results: dict[int, dict[str, float]]) -> str:
    lines = [f"{'K':>3} {'n_tr':>6} {'ev_auc':>7} {'ev_acc':>7} {'ha_auc':>7} {'ha_acc':>7} {'f1_r':>7} {'f1_mse':>7}"]
    for k in sorted(results.keys()):
        m = results[k]
        if m.get("n_traces", 0) == 0:
            lines.append(f"{k:>3} {'--':>6} (no val cases at this K)")
            continue
        lines.append(
            f"{k:>3} {m['n_traces']:>6} {m['evidence_auc']:>7.3f} {m['evidence_acc']:>7.3f} "
            f"{m['hop_answer_auc']:>7.3f} {m['hop_answer_acc']:>7.3f} "
            f"{m['final_f1_pearson']:>7.3f} {m['final_f1_mse']:>7.3f}"
        )
    return "\n".join(lines)


def format_per_k_all_vs_natural_table(results: dict[int, dict[str, dict[str, float]]]) -> str:
    header = f"{'K':>3} {'view':>8} {'n_tr':>6} {'ev_auc':>7} {'ev_acc':>7} {'ha_auc':>7} {'ha_acc':>7} {'f1_r':>7} {'f1_mse':>7}"
    lines = [header]
    for k in sorted(results.keys()):
        lines.append(_fmt_row(k, "all", results[k]["all"]))
        lines.append(_fmt_row(k, "natural", results[k]["natural"]))
    return "\n".join(lines)
