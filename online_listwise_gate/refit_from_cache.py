#!/usr/bin/env python3
"""
Fast re-fit of the listwise gate from a feature cache written by
train_online_wavefront.py --cache-features-dir <dir> -- skips retrieval, Llama hidden-state
extraction, target-judge LLM calls, and short-answer generation entirely (the expensive,
hours-scale part of a full run) and re-runs ONLY the cheap part (PCA fit + incremental model
training, seconds-to-minutes) against the cached raw feature arrays.

This is deliberately a SCREENING tool, not a substitute for a full run:

  hop 1's cached data is identical no matter which model/hyperparameters trained it (hop 1
  has no prior committed pick to depend on), so it's always safe to reuse. hop 2+ in the
  original run depended on THAT run's own model's real committed picks (DAgger-style
  self-propagation, see train_online_wavefront.py's module docstring) -- so a sweep of
  --pca-dim / --model-type / --include-h-prev refit from one cache compares those choices on
  ONE FIXED trajectory (whichever run produced the cache), not each choice's own from-scratch
  online trajectory. That is a reasonable, much faster proxy for "does more PCA capacity /
  a bigger scorer / an extra history feature help fit this data better" -- which is exactly
  the kind of question a --pca-dim or --model-type sweep is asking -- but it is NOT a
  from-scratch DAgger run, so whichever config wins here should be re-run through
  train_online_wavefront.py directly (no cache) for a real end-to-end confirmation before
  being treated as the new default.

Usage:
  python refit_from_cache.py --cache-dir results/<run>/feature_cache --pca-dim 128
  python refit_from_cache.py --cache-dir results/<run>/feature_cache --pca-dim 128 \
      --model-type mlp --include-h-prev
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import (  # noqa: E402
    FeaturePCA, ListwiseGateModel, MLPGateModel, groups_accuracy, listwise_loss, save_artifact,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True,
                     help="Directory written by train_online_wavefront.py --cache-features-dir.")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--model-type", choices=["bilinear", "mlp"], default="bilinear")
    ap.add_argument("--mlp-hidden-dim", type=int, default=128)
    ap.add_argument("--include-h-prev", action="store_true", default=False,
                     help="Requires the cache to have been written with --include-h-prev too "
                          "(errors out clearly if the cached hprev array is empty).")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-epochs-per-hop", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0",
                     help="train_online_wavefront.py hardcodes CPU here since PCA'd features "
                          "are small (a few hundred dims) and its per-hop pool tops out around "
                          "10k groups; this script's accumulated pool can reach ~25k groups by "
                          "hop 4 and listwise_loss() runs one Python-level forward pass per "
                          "group per epoch (not batched), so GPU can still help there, "
                          "especially for --model-type mlp with a wider --mlp-hidden-dim. Falls "
                          "back to cpu with a warning if CUDA isn't actually available.")
    ap.add_argument("--results-root", type=Path, default=HERE / "results")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--out-artifact", type=Path, default=None)
    return ap.parse_args()


def load_hop_cache(cache_dir: Path, hop_j: int, include_h_prev: bool) -> dict[str, Any]:
    npz_path = cache_dir / f"hop{hop_j}.npz"
    meta_path = cache_dir / f"hop{hop_j}_meta.json"
    if not npz_path.exists():
        raise SystemExit(f"missing {npz_path} -- was --cache-features-dir set on the run that "
                          f"produced {cache_dir}, and did it reach hop {hop_j}?")
    z = np.load(npz_path)
    hprev = None
    if include_h_prev:
        if z["hprev"].size == 0:
            raise SystemExit(f"--include-h-prev requested but {npz_path} has an empty hprev "
                              f"array -- the cache was written without --include-h-prev on "
                              f"train_online_wavefront.py, re-run it with that flag to get one.")
        hprev = z["hprev"].astype(np.float32)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
    group_bounds = [
        (int(s), int(e), None if g < 0 else int(g))
        for s, e, g in zip(z["starts"], z["ends"], z["golds"])
    ]
    return {
        "q": z["q"].astype(np.float32), "d": z["d"].astype(np.float32),
        "h": z["h"].astype(np.float32), "delta": z["delta"].astype(np.float32),
        "hprev": hprev, "group_bounds": group_bounds, "meta": meta,
    }


def main() -> None:
    run_t0 = time.perf_counter()
    args = parse_args()
    if args.include_h_prev and args.model_type != "mlp":
        raise SystemExit("--include-h-prev requires --model-type mlp (same constraint as "
                          "train_online_wavefront.py)")

    max_hops = 0
    while (args.cache_dir / f"hop{max_hops + 1}.npz").exists():
        max_hops += 1
    if max_hops == 0:
        raise SystemExit(f"no hop*.npz files found under {args.cache_dir}")
    print(f"Found cached hops 1..{max_hops} in {args.cache_dir}")

    tag = args.run_tag or "refit_from_cache"
    out_dir = args.out_dir or (args.results_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_artifact = args.out_artifact or (out_dir / "artifact.joblib")
    print(f"Output directory: {out_dir}")

    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    if "cuda" in args.device and device.type == "cpu":
        print(f"[warn] --device {args.device} requested but CUDA isn't available -- falling back to cpu", flush=True)
    print(f"Using device: {device}")
    feature_pca: FeaturePCA | None = None
    model: ListwiseGateModel | MLPGateModel | None = None
    optimizer: torch.optim.Optimizer | None = None
    accumulated_train_groups: list[tuple[torch.Tensor, int]] = []
    val_groups_by_hop: dict[int, list[tuple[torch.Tensor, int]]] = {}
    per_hop_stats: list[dict] = []

    for hop_j in range(1, max_hops + 1):
        t0 = time.perf_counter()
        print(f"\n=== hop {hop_j}/{max_hops} (from cache) ===", flush=True)
        cached = load_hop_cache(args.cache_dir, hop_j, args.include_h_prev)
        group_bounds, meta = cached["group_bounds"], cached["meta"]
        n_active = len(group_bounds)
        print(f"[hop {hop_j}] {n_active} examples (cached)", flush=True)

        if feature_pca is None:
            n_branches = 5 if args.include_h_prev else 4
            print(f"[hop {hop_j}] fitting the {n_branches} PCA transforms on hop-1 data "
                  f"({cached['q'].shape[0]} candidates) ...")
            feature_pca = FeaturePCA(pca_dim=args.pca_dim, include_h_prev=args.include_h_prev)
            feature_pca.fit(cached["q"], cached["d"], cached["h"], cached["delta"], cached["hprev"])
            if args.model_type == "mlp":
                model = MLPGateModel(feature_pca.out_dim, hidden_dim=args.mlp_hidden_dim)
            else:
                model = ListwiseGateModel(*feature_pca.branch_dims)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        transformed = feature_pca.transform(cached["q"], cached["d"], cached["h"], cached["delta"], cached["hprev"])
        transformed_t = torch.tensor(transformed, dtype=torch.float32, device=device)

        new_groups: list[tuple[torch.Tensor, int]] = []
        group_feats_by_ex: dict[int, torch.Tensor] = {}
        for i, (start, end, gold_local) in enumerate(group_bounds):
            feats = transformed_t[start:end]
            group_feats_by_ex[i] = feats
            if gold_local is not None and end > start:
                new_groups.append((feats, gold_local))

        rng = random.Random(args.sample_seed * 1000 + hop_j)
        shuffled = new_groups[:]
        rng.shuffle(shuffled)
        n_val = round(len(shuffled) * args.val_frac)
        new_val_groups, new_train_groups = shuffled[:n_val], shuffled[n_val:]
        accumulated_train_groups.extend(new_train_groups)
        val_groups_by_hop[hop_j] = new_val_groups
        n_val_total = sum(len(g) for g in val_groups_by_hop.values())
        print(f"[hop {hop_j}] +{len(new_train_groups)} train / +{len(new_val_groups)} val groups "
              f"(gold present in pool), {len(accumulated_train_groups)}/{n_val_total} "
              f"accumulated total", flush=True)

        best_val_metric = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        epochs_since_improve = 0
        epochs_run = 0
        if accumulated_train_groups:
            model.train()
            for epoch in range(args.max_epochs_per_hop):
                optimizer.zero_grad()
                loss = listwise_loss(model, accumulated_train_groups)
                loss.backward()
                optimizer.step()
                epochs_run = epoch + 1

                model.eval()
                hop_accs = [groups_accuracy(model, g) for g in val_groups_by_hop.values() if g]
                val_metric = sum(hop_accs) / len(hop_accs) if hop_accs else groups_accuracy(model, accumulated_train_groups)
                model.train()

                if val_metric > best_val_metric + 1e-4:
                    best_val_metric = val_metric
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= args.patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            print(f"[hop {hop_j}] trained {epochs_run} epochs (early-stopped, patience="
                  f"{args.patience}) on {len(accumulated_train_groups)} train groups, "
                  f"best mean-per-hop val_accuracy={best_val_metric:.4f} over {len(val_groups_by_hop)} "
                  f"hops' validation sets ({n_val_total} groups total)", flush=True)

        # No "commit real pick" step here (that step exists in train_online_wavefront.py only
        # to decide what hop_{j+1}'s retrieval/context looks like -- moot here since hop_{j+1}'s
        # data is already fixed in the cache). Still report pick accuracy the same way, purely
        # as a monitoring number (same semantics as train_online_wavefront.py's this-hop
        # pick_accuracy: does argmax(model(feats)) match the cached gold_local).
        model.eval()
        n_correct = 0
        n_correct_by_src: dict[tuple[str, str], int] = {}
        n_total_by_src: dict[tuple[str, str], int] = {}
        n_scored = 0
        with torch.no_grad():
            for i, (start, end, gold_local) in enumerate(group_bounds):
                if end <= start or gold_local is None:
                    continue
                n_scored += 1
                pick = int(torch.argmax(model(group_feats_by_ex[i])).item())
                src_key = (meta[i]["dataset"], meta[i]["decompose_mode"]) if i < len(meta) else ("?", "?")
                n_total_by_src[src_key] = n_total_by_src.get(src_key, 0) + 1
                if pick == gold_local:
                    n_correct += 1
                    n_correct_by_src[src_key] = n_correct_by_src.get(src_key, 0) + 1
        hop_acc = n_correct / n_scored if n_scored else 0.0
        acc_by_src = {
            f"{ds}/{mode}": round(n_correct_by_src.get((ds, mode), 0) / n, 4)
            for (ds, mode), n in n_total_by_src.items() if n
        }
        print(f"[hop {hop_j}] this-hop pick accuracy (model's own picks vs cached gold): "
              f"{hop_acc:.4f}  by source: {acc_by_src}", flush=True)

        elapsed = time.perf_counter() - t0
        per_hop_stats.append({
            "hop": hop_j, "n_active": n_active,
            "n_train_groups_added": len(new_train_groups), "n_val_groups_added": len(new_val_groups),
            "n_train_groups_total": len(accumulated_train_groups), "n_val_groups_total": n_val_total,
            "epochs_run": epochs_run,
            "best_val_accuracy": round(best_val_metric, 4) if best_state is not None else None,
            "pick_accuracy": round(hop_acc, 4), "pick_accuracy_by_source": acc_by_src,
            "elapsed_min": round(elapsed / 60, 2),
        })
        print(f"[hop {hop_j}] elapsed {elapsed / 60:.2f} min", flush=True)

    model = model.cpu()  # artifacts stay CPU-portable regardless of --device, matching
    # train_online_wavefront.py's saved artifacts and what downstream eval scripts expect to
    # load without needing CUDA available at load time.
    save_artifact(args.out_artifact, feature_pca, model, meta={
        "pca_dim": args.pca_dim, "model_type": args.model_type,
        "include_h_prev": args.include_h_prev,
        "refit_from_cache": str(args.cache_dir),
        "per_hop_stats": per_hop_stats,
    })
    run_wall_sec = time.perf_counter() - run_t0
    (out_dir / "run_meta.json").write_text(json.dumps({
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "per_hop_stats": per_hop_stats,
        "wall_clock_hr": round(run_wall_sec / 3600, 3),
        "artifact": str(args.out_artifact),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved frozen artifact -> {args.out_artifact}")
    print(f"Run meta -> {out_dir / 'run_meta.json'}")
    print(f"Wall clock hr {run_wall_sec / 3600:.4f}")


if __name__ == "__main__":
    main()
