#!/usr/bin/env python3
"""Full training loop for MTLGateModel on the packed feature cache. Case-level train/val split,
stratified by K (2/3/4-hop) so a case's recipe1-4 variants never straddle the split. Validation
is reported broken down by K at every eval interval, and the final report is per-K too.

Usage:
  python train.py --cache cache/musique_train.npz --epochs 20 --val-ratio 0.1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from model import MTLGateModel, mtl_loss
from data_split import stratified_case_split
from packed_data import PackedCache, iter_epoch_batches
from evaluate import evaluate_per_k_all_and_natural, format_per_k_all_vs_natural_table


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("cache/musique_train.npz"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                     help="AdamW weight decay (0 = plain Adam, the original behavior)")
    ap.add_argument("--dropout", type=float, default=0.1, help="dropout inside the shared trunk")
    ap.add_argument("--trunk-hidden", type=int, default=512)
    ap.add_argument("--trunk-out", type=int, default=128)
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--patience", type=int, default=0,
                     help="early stopping: stop after this many epochs with no val-loss "
                          "improvement, and keep the best-val-loss weights (0 = disabled, "
                          "train the full --epochs and keep the last weights)")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--traces-path", type=Path,
                     default=Path("../new_data_recipe/output_full/traces_v2/musique/train_with_hop_labels.jsonl"),
                     help="only used to pull is_natural_seed flags for the natural-vs-all eval breakdown")
    ap.add_argument("--save-path", type=Path, default=Path("checkpoints/mtl_gate.pt"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)

    print(f"loading cache {args.cache} ...")
    t0 = time.perf_counter()
    cache = PackedCache(args.cache)
    print(f"  {cache.n_traces} traces, x_all {cache.x_all.shape}, {time.perf_counter() - t0:.1f}s")
    cache.load_natural_seed_flags(args.traces_path)
    print(f"  is_natural_seed=True: {int(cache.is_natural_seed.sum())}/{cache.n_traces} traces")

    train_case_ids, val_case_ids = stratified_case_split(
        list(cache.case_id), list(cache.K), val_ratio=args.val_ratio, seed=args.split_seed,
    )
    print(f"case split: {len(train_case_ids)} train cases, {len(val_case_ids)} val cases")

    train_idx = cache.indices_for_case_ids(train_case_ids)
    print(f"train traces: {len(train_idx)} (of {cache.n_traces})")

    val_idx = cache.indices_for_case_ids(val_case_ids)

    D = cache.x_all.shape[1]
    model = MTLGateModel(input_dim=D, trunk_hidden=args.trunk_hidden, trunk_out=args.trunk_out,
                          head_hidden=args.head_hidden, dropout=args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}  (trunk {args.trunk_hidden}->{args.trunk_out}, "
          f"dropout={args.dropout}, weight_decay={args.weight_decay}, patience={args.patience})\n")

    @torch.no_grad()
    def val_loss() -> tuple[float, dict[str, float]]:
        """Held-out loss, same objective as training -- what early stopping tracks. Reported
        alongside the train loss so the train/val gap (i.e. the overfitting) is visible per
        epoch instead of only at the end."""
        model.eval()
        tot, parts, nb = 0.0, {"loss_evidence": 0.0, "loss_hop_answer": 0.0, "loss_final_f1": 0.0}, 0
        for b in iter_epoch_batches(cache, val_idx, args.batch_size, shuffle=False):
            out = model(b["x"])
            ls = mtl_loss(out, evidence_label=b["evidence_label"],
                           hop_answer_label=b["hop_answer_label"],
                           final_f1_label=b["final_f1_label"], mask=b["mask"])
            tot += ls["loss_total"].item()
            for k in parts:
                parts[k] += ls[k].item()
            nb += 1
        model.train()
        return tot / max(nb, 1), {k: v / max(nb, 1) for k, v in parts.items()}

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_best = 0

    for epoch in range(args.epochs):
        t_epoch = time.perf_counter()
        model.train()
        total_loss, n_batches = 0.0, 0
        loss_parts = {"loss_evidence": 0.0, "loss_hop_answer": 0.0, "loss_final_f1": 0.0}
        for batch in iter_epoch_batches(cache, train_idx, args.batch_size, shuffle=True, seed=epoch):
            out = model(batch["x"])
            losses = mtl_loss(
                out, evidence_label=batch["evidence_label"], hop_answer_label=batch["hop_answer_label"],
                final_f1_label=batch["final_f1_label"], mask=batch["mask"],
            )
            opt.zero_grad()
            losses["loss_total"].backward()
            opt.step()
            total_loss += losses["loss_total"].item()
            for k in loss_parts:
                loss_parts[k] += losses[k].item()
            n_batches += 1

        avg = total_loss / n_batches
        avg_parts = {k: v / n_batches for k, v in loss_parts.items()}
        v_avg, v_parts = val_loss()
        dt = time.perf_counter() - t_epoch
        print(f"epoch {epoch:3d} | train={avg:.4f} val={v_avg:.4f} gap={v_avg - avg:+.4f} "
              f"| val(ev={v_parts['loss_evidence']:.4f} ha={v_parts['loss_hop_answer']:.4f} "
              f"f1={v_parts['loss_final_f1']:.4f}) | {dt:.1f}s")

        if v_avg < best_val - 1e-5:
            best_val, best_epoch, epochs_since_best = v_avg, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_since_best += 1
            if args.patience and epochs_since_best >= args.patience:
                print(f"early stop: no val improvement for {args.patience} epoch(s); "
                      f"best was epoch {best_epoch} (val={best_val:.4f})")
                break

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            results = evaluate_per_k_all_and_natural(model, cache, val_case_ids)
            print(format_per_k_all_vs_natural_table(results))
            print()

    if args.patience and best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best-val weights from epoch {best_epoch} (val={best_val:.4f})")

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "input_dim": D,
                "trunk_hidden": args.trunk_hidden, "trunk_out": args.trunk_out,
                "head_hidden": args.head_hidden}, args.save_path)
    print(f"saved checkpoint -> {args.save_path}")

    print("\n=== final validation, per K (all traces vs. recipe2-natural-only) ===")
    results = evaluate_per_k_all_and_natural(model, cache, val_case_ids)
    print(format_per_k_all_vs_natural_table(results))


if __name__ == "__main__":
    main()
