#!/usr/bin/env python3
"""Diagnostic: what does the trained MTL gate actually output? Checks whether the three heads
produce a real spread of probabilities or have collapsed to one value, and whether that spread
actually separates the positive from the negative label group. Runs on the held-out val split
(same case-level stratified split train.py uses), CPU only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import MTLGateModel  # noqa: E402
from data_split import stratified_case_split  # noqa: E402
from packed_data import PackedCache, batch_from_indices  # noqa: E402

PCTS = [0, 1, 10, 25, 50, 75, 90, 99, 100]


def describe(name: str, v: np.ndarray) -> None:
    q = np.percentile(v, PCTS)
    print(f"{name:<14} n={len(v):>7}  mean={v.mean():.3f} std={v.std():.3f}  "
          + "  ".join(f"p{p}={x:.3f}" for p, x in zip(PCTS, q)))


def buckets(name: str, v: np.ndarray) -> None:
    edges = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    counts = [(v >= lo) & (v < hi) if hi < 1.0 else (v >= lo) & (v <= hi)
              for lo, hi in zip(edges[:-1], edges[1:])]
    line = "  ".join(f"[{lo:.1f},{hi:.1f})={c.mean() * 100:5.1f}%"
                     for lo, hi, c in zip(edges[:-1], edges[1:], counts))
    print(f"{name:<14} {line}")


def collect(model, cache, idx, batch_size: int = 64):
    preds = {k: [] for k in ("evidence", "hop_answer", "final_f1")}
    labels = {k: [] for k in ("evidence", "hop_answer", "final_f1")}
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b = batch_from_indices(cache, idx[s:s + batch_size])
            out = model(b["x"])
            m = b["mask"].bool()
            for key, logit_key, label_key in [
                ("evidence", "evidence_logit", "evidence_label"),
                ("hop_answer", "hop_answer_logit", "hop_answer_label"),
                ("final_f1", "final_f1_logit", "final_f1_label"),
            ]:
                preds[key].append(torch.sigmoid(out[logit_key])[m].numpy())
                labels[key].append(b[label_key][m].numpy())
    return ({k: np.concatenate(v) for k, v in preds.items()},
            {k: np.concatenate(v) for k, v in labels.items()})


def main() -> None:
    from sklearn.metrics import roc_auc_score

    torch.set_num_threads(4)
    cache = PackedCache(HERE / "cache/musique_train.npz")
    train_cases, val_cases = stratified_case_split(
        list(cache.case_id), list(cache.K), val_ratio=0.1, seed=42)
    val_idx = cache.indices_for_case_ids(val_cases)
    train_idx = cache.indices_for_case_ids(train_cases)
    # subsample train to val size so the two sides are comparable in cost
    rng = np.random.default_rng(0)
    train_idx = rng.choice(train_idx, size=min(len(val_idx), len(train_idx)), replace=False)
    print(f"train(subsampled): {len(train_idx)} traces   val: {len(val_idx)} traces\n")

    ckpt = torch.load(HERE / "checkpoints/mtl_gate.pt", map_location="cpu")
    model = MTLGateModel(
        input_dim=ckpt["input_dim"],
        trunk_hidden=ckpt.get("trunk_hidden", 512),
        trunk_out=ckpt.get("trunk_out", 128),
        head_hidden=ckpt.get("head_hidden", 64),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tr_p, tr_y = collect(model, cache, train_idx)
    va_p, va_y = collect(model, cache, val_idx)

    print("=== train vs val（看有没有过拟合 gap）===")
    print(f"{'head':<12} {'split':<6} {'AUC':>7} {'acc@0.5':>8} {'BCE':>7} {'pred_mean':>10} {'label_mean':>11}")
    for key in ("evidence", "hop_answer", "final_f1"):
        for split, p, y in (("train", tr_p[key], tr_y[key]), ("val", va_p[key], va_y[key])):
            yb = (y > 0.5).astype(np.float32)
            auc = roc_auc_score(yb, p) if len(np.unique(yb)) > 1 else float("nan")
            acc = ((p > 0.5) == (yb > 0.5)).mean()
            eps = 1e-7
            bce = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()
            print(f"{key:<12} {split:<6} {auc:>7.4f} {acc:>8.4f} {bce:>7.4f} "
                  f"{p.mean():>10.3f} {y.mean():>11.3f}")
    print()

    for key in va_p:
        p, y = va_p[key], va_y[key]
        print(f"=== {key} head（验证集）===")
        describe("预测值整体", p)
        buckets("预测值分桶", p)
        pos, neg = p[y > 0.5], p[y <= 0.5]
        if len(pos):
            describe("  标签为正", pos)
        if len(neg):
            describe("  标签为负", neg)
        print(f"  标签分布: 正 {(y > 0.5).mean():.1%} / 负 {(y <= 0.5).mean():.1%}"
              f"   标签本身 mean={y.mean():.3f}")
        print()


if __name__ == "__main__":
    main()
