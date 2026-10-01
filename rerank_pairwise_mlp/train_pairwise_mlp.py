#!/usr/bin/env python3
"""
Scheme A: train a small MLP scorer on [emb_score, PCA(Delta), hop features]
with a PAIRWISE ranking loss (RankNet-style logistic loss on (correct,
wrong) pairs from the same hop) instead of pointwise classification and
instead of the fixed emb_score - lambda*abnormal_score formula. See
update_doc/0713/0713update.md's rerank-structure discussion for why this is
a different structure, not just a re-weighting.

Hop features (one-hot(j) over the 4 pooled transition roles + raw K) are
included because production's gate is 4 SEPARATE models, one per pooled j
(gate/fit_lr_gate_pooled.py), each free to use its own layer — this script
originally pooled every (K, j) together into ONE model with no notion of
"which hop this is", unlike production. Rather than fragmenting training
data into 4 separate models (which would also need per-j Delta extraction
at different layers to match production), the cheaper fix tried here is
letting the single shared model condition on hop position explicitly.

Input: the .npz produced by build_training_data.py (deltas, emb_scores,
labels, pair_id, K, j) for train (required) and dev (optional, for
held-out pairwise accuracy).

Output: artifacts/{pca.joblib, model.pt, meta.json} — pca.joblib is a
scikit-learn PCA fit on train deltas only; model.pt is the trained MLP's
state_dict + architecture config.

Usage:
  python train_pairwise_mlp.py --train-npz data/musique_train_pairwise.npz \
      --dev-npz data/musique_dev_pairwise.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
NUM_J_CLASSES = 4  # pooled transition roles: j=0 Q->E1, 1 E1->E2, 2 E2->E3, 3 E3->E4


class PairwiseScorer(nn.Module):
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
    return d["deltas"], d["emb_scores"], d["labels"], d["pair_id"], d["K"], d["j"]


def hop_features(K: np.ndarray, j: np.ndarray) -> np.ndarray:
    """one-hot(j, 4 classes) + raw K, i.e. the hop-depth info production's
    4-separate-models gate gets "for free" from not being pooled."""
    onehot = np.zeros((len(j), NUM_J_CLASSES), dtype=np.float32)
    onehot[np.arange(len(j)), np.clip(j, 0, NUM_J_CLASSES - 1)] = 1.0
    return np.concatenate([onehot, K.reshape(-1, 1).astype(np.float32)], axis=1)


def build_pairs(deltas, emb_scores, labels, pair_id):
    """Group rows by pair_id into (correct_row_idx, wrong_row_idx) pairs."""
    by_pair: dict[str, dict[int, int]] = {}
    for i, (pid, lab) in enumerate(zip(pair_id, labels)):
        by_pair.setdefault(pid, {})[int(lab)] = i
    pairs = []
    for pid, d in by_pair.items():
        if 1 in d and 0 in d:
            pairs.append((d[1], d[0]))
    return pairs


def features(deltas: np.ndarray, emb_scores: np.ndarray, K: np.ndarray, j: np.ndarray, pca) -> np.ndarray:
    z = pca.transform(deltas.astype(np.float64))
    hf = hop_features(K, j)
    return np.concatenate([emb_scores.reshape(-1, 1), z, hf], axis=1).astype(np.float32)


def pairwise_accuracy(model: PairwiseScorer, X: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(X)).numpy()
    correct = sum(1 for i, j in pairs if scores[i] > scores[j])
    return correct / len(pairs) if pairs else 0.0


def baseline_pairwise_accuracy(emb_scores: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    """emb_score alone (lambda=0 formula, i.e. pure cosine similarity ranking) as a sanity floor."""
    correct = sum(1 for i, j in pairs if emb_scores[i] > emb_scores[j])
    return correct / len(pairs) if pairs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-npz", type=Path, required=True)
    ap.add_argument("--dev-npz", type=Path, default=None)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=HERE / "artifacts")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from sklearn.decomposition import PCA

    train_deltas, train_emb, train_labels, train_pid, train_K, train_j = load_npz(args.train_npz)
    print(f"Train rows: {len(train_deltas)}")

    pca = PCA(n_components=args.pca_dim)
    pca.fit(train_deltas.astype(np.float64))

    X_train = features(train_deltas, train_emb, train_K, train_j, pca)
    train_pairs = build_pairs(train_deltas, train_emb, train_labels, train_pid)
    print(f"Train pairs: {len(train_pairs)}")

    dev_data = None
    if args.dev_npz is not None:
        dev_deltas, dev_emb, dev_labels, dev_pid, dev_K, dev_j = load_npz(args.dev_npz)
        X_dev = features(dev_deltas, dev_emb, dev_K, dev_j, pca)
        dev_pairs = build_pairs(dev_deltas, dev_emb, dev_labels, dev_pid)
        dev_data = (X_dev, dev_emb, dev_pairs)
        print(f"Dev rows: {len(dev_deltas)}  dev pairs: {len(dev_pairs)}")

    model = PairwiseScorer(in_dim=X_train.shape[1], hidden_dim=args.hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    X_train_t = torch.from_numpy(X_train)

    idx_correct = torch.tensor([p[0] for p in train_pairs], dtype=torch.long)
    idx_wrong = torch.tensor([p[1] for p in train_pairs], dtype=torch.long)

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        scores = model(X_train_t)
        margin = scores[idx_correct] - scores[idx_wrong]
        loss = -torch.nn.functional.logsigmoid(margin).mean()
        loss.backward()
        opt.step()

        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            train_acc = pairwise_accuracy(model, X_train, train_pairs)
            msg = f"epoch {epoch:3d}  loss={loss.item():.4f}  train_pairwise_acc={train_acc:.4f}"
            if dev_data is not None:
                X_dev, _dev_emb, dev_pairs = dev_data
                dev_acc = pairwise_accuracy(model, X_dev, dev_pairs)
                msg += f"  dev_pairwise_acc={dev_acc:.4f}"
            print(msg)

    if dev_data is not None:
        _X_dev, dev_emb, dev_pairs = dev_data
        base_acc = baseline_pairwise_accuracy(dev_emb, dev_pairs)
        print(f"\n[baseline] emb_score-only pairwise accuracy on dev: {base_acc:.4f}")
        print(f"[model]    trained MLP pairwise accuracy on dev:      "
              f"{pairwise_accuracy(model, dev_data[0], dev_pairs):.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(pca, args.out_dir / "pca.joblib")
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": X_train.shape[1],
        "hidden_dim": args.hidden_dim,
    }, args.out_dir / "model.pt")
    meta = {
        "scheme": "pairwise_mlp", "pca_dim": args.pca_dim, "hidden_dim": args.hidden_dim,
        "epochs": args.epochs, "lr": args.lr, "n_train_pairs": len(train_pairs),
        "n_dev_pairs": len(dev_data[2]) if dev_data else None,
        "layer": 31,  # hidden_states/{split}/activations/ is single-layer, always layer 31
        "hop_features": True, "num_j_classes": NUM_J_CLASSES,  # eval scripts must build the same [emb, PCA(delta), onehot(j), K]
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved artifacts -> {args.out_dir}")


if __name__ == "__main__":
    main()
