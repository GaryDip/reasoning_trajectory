#!/usr/bin/env python3
"""
Scheme B: train a small MLP scorer on [emb_score, PCA(Delta), hop features]
with a LISTWISE softmax cross-entropy loss over each hop's REAL top-k
retrieved pool (from build_topk_pools.py + extract_pool_hidden_states.py) —
the gold candidate is the target class among however many candidates were
actually retrieved for that hop; pools with no gold candidate present are
excluded (nothing to supervise). This is the "contrastive" structure
discussed in update_doc/0713/0713update.md: the model has to win against
the SAME competitors it will actually face at inference, not one
arbitrarily sampled distractor (contrast with rerank_pairwise_mlp's
pairwise scheme).

The scorer architecture itself is the same shape as Scheme A's (independent
per-candidate MLP on [emb_score, PCA(Delta), hop features]) — what's
different is the training objective and the realism of the training data,
not attention between candidates (that would be a further step, not built
here). Hop features (one-hot(j) + K) are included for the same reason as
Scheme A: production's gate is 4 separate models, one per pooled j; this
script pools everything into one model, so hop position is passed in
explicitly instead.

Usage:
  python train_listwise_ranker.py --train-npz data/musique_train_pools.hidden.npz \
      --dev-npz data/musique_dev_pools.hidden.npz
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


NUM_J_CLASSES = 4  # pooled transition roles: j=0 Q->E1, 1 E1->E2, 2 E2->E3, 3 E3->E4


class ListwiseScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    return d["deltas"], d["emb_scores"], d["is_gold"], d["pool_id"], d["K"], d["j"]


def group_pools(pool_id: np.ndarray, is_gold: np.ndarray) -> list[tuple[str, list[int], int | None]]:
    """Returns (pool_id, row_indices, gold_local_index_or_None) per pool."""
    by_pool: dict[str, list[int]] = defaultdict(list)
    for i, pid in enumerate(pool_id):
        by_pool[pid].append(i)
    groups = []
    for pid, idxs in by_pool.items():
        gold_local = next((k for k, i in enumerate(idxs) if is_gold[i]), None)
        groups.append((pid, idxs, gold_local))
    return groups


def hop_features(K: np.ndarray, j: np.ndarray) -> np.ndarray:
    onehot = np.zeros((len(j), NUM_J_CLASSES), dtype=np.float32)
    onehot[np.arange(len(j)), np.clip(j, 0, NUM_J_CLASSES - 1)] = 1.0
    return np.concatenate([onehot, K.reshape(-1, 1).astype(np.float32)], axis=1)


def features(deltas: np.ndarray, emb_scores: np.ndarray, K: np.ndarray, j: np.ndarray, pca) -> np.ndarray:
    z = pca.transform(deltas.astype(np.float64))
    hf = hop_features(K, j)
    return np.concatenate([emb_scores.reshape(-1, 1), z, hf], axis=1).astype(np.float32)


def top1_accuracy(scores: np.ndarray, groups: list[tuple[str, list[int], int | None]]) -> float:
    n_supervised = 0
    n_correct = 0
    for _pid, idxs, gold_local in groups:
        if gold_local is None:
            continue
        n_supervised += 1
        local_scores = scores[idxs]
        if int(np.argmax(local_scores)) == gold_local:
            n_correct += 1
    return n_correct / n_supervised if n_supervised else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-npz", type=Path, required=True)
    ap.add_argument("--dev-npz", type=Path, default=None)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from sklearn.decomposition import PCA

    train_deltas, train_emb, train_gold, train_pid, train_K, train_j = load_npz(args.train_npz)
    train_groups = group_pools(train_pid, train_gold)
    n_supervised_train = sum(1 for _p, _i, g in train_groups if g is not None)
    print(f"Train: {len(train_deltas)} rows, {len(train_groups)} pools "
          f"({n_supervised_train} with gold present)")

    pca = PCA(n_components=args.pca_dim)
    pca.fit(train_deltas.astype(np.float64))
    X_train = features(train_deltas, train_emb, train_K, train_j, pca)

    dev_data = None
    if args.dev_npz is not None:
        dev_deltas, dev_emb, dev_gold, dev_pid, dev_K, dev_j = load_npz(args.dev_npz)
        dev_groups = group_pools(dev_pid, dev_gold)
        X_dev = features(dev_deltas, dev_emb, dev_K, dev_j, pca)
        n_supervised_dev = sum(1 for _p, _i, g in dev_groups if g is not None)
        print(f"Dev: {len(dev_deltas)} rows, {len(dev_groups)} pools "
              f"({n_supervised_dev} with gold present)")
        dev_data = (X_dev, dev_emb, dev_groups)

    model = ListwiseScorer(in_dim=X_train.shape[1], hidden_dim=args.hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    X_train_t = torch.from_numpy(X_train)

    supervised_groups = [(idxs, g) for _p, idxs, g in train_groups if g is not None]

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        all_scores = model(X_train_t)
        losses = []
        for idxs, gold_local in supervised_groups:
            local_scores = all_scores[torch.tensor(idxs, dtype=torch.long)]
            target = torch.tensor(gold_local, dtype=torch.long)
            losses.append(torch.nn.functional.cross_entropy(local_scores.unsqueeze(0), target.unsqueeze(0)))
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                train_scores = model(X_train_t).numpy()
            train_acc = top1_accuracy(train_scores, train_groups)
            msg = f"epoch {epoch:3d}  loss={loss.item():.4f}  train_top1_acc={train_acc:.4f}"
            if dev_data is not None:
                X_dev, _dev_emb, dev_groups = dev_data
                with torch.no_grad():
                    dev_scores = model(torch.from_numpy(X_dev)).numpy()
                dev_acc = top1_accuracy(dev_scores, dev_groups)
                msg += f"  dev_top1_acc={dev_acc:.4f}"
            print(msg)

    if dev_data is not None:
        _X_dev, dev_emb, dev_groups = dev_data
        base_acc = top1_accuracy(dev_emb, dev_groups)  # cosine-only baseline
        print(f"\n[baseline] emb_score-only top-1 accuracy on dev: {base_acc:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(pca, args.out_dir / "pca.joblib")
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": X_train.shape[1],
        "hidden_dim": args.hidden_dim,
    }, args.out_dir / "model.pt")
    meta = {
        "scheme": "listwise_contrastive", "pca_dim": args.pca_dim, "hidden_dim": args.hidden_dim,
        "epochs": args.epochs, "lr": args.lr, "n_train_pools_supervised": len(supervised_groups),
        "layer": 31,
        "hop_features": True, "num_j_classes": NUM_J_CLASSES,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved artifacts -> {args.out_dir}")


if __name__ == "__main__":
    main()
