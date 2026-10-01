#!/usr/bin/env python3
"""
The listwise reranker itself: a score over four INDEPENDENTLY-PCA'd feature branches per
candidate --

    PCA(query BGE embedding), PCA(candidate-passage BGE embedding),
    PCA(h_after), PCA(delta_h)

-- trained with softmax cross-entropy over each hop's own candidate pool (the gold
candidate's index is the target logit), i.e. plain listwise softmax ranking (ListNet-style):
whichever candidate the model assigns the highest score to should be the gold one.

score_qd = q^T W_qd d       (bilinear interaction: does this query "match" this passage?)
score_hd = h^T W_hd delta   (bilinear interaction: does the representation shift this candidate
                             causes line up with the accumulated context?)
g        = sigmoid(w_gate^T [q, d, h, delta] + b_gate)   (data-dependent mixing weight)
score    = g * score_qd + (1 - g) * score_hd

Both pairs get an explicit BILINEAR interaction term (nn.Bilinear), not a plain linear term --
a plain Linear layer over a concatenation can only learn independent per-coordinate weights for
each vector; it cannot represent "how aligned these two vectors are for this specific pair"
(dot-product/interaction terms are not expressible as a linear combination of concatenated
coordinates). (q, d) needs this because query/passage relevance is inherently a pairwise
comparison (this is exactly what cosine similarity used to compute directly: q . d) -- the
whole point of dropping the pre-computed cosine scalar was to let the model learn how to
compare q and d itself, so the scorer needs a term that can actually represent a comparison.
(h, delta) gets the same treatment for a matching reason: delta is literally h_after - h_prev,
so whether this candidate's representation shift is consistent with the accumulated context is
also a pairwise comparison, not two independent scalars to just add up.

The two bilinear scores are then combined with a learned, per-candidate GATE (not a fixed
mixing weight) -- g depends on all four raw feature branches, so the model can decide, per
candidate, whether to lean more on "does the query semantically match this passage" or "does
picking this passage make sense given the accumulated reasoning so far", rather than always
weighting the two signals the same way.

This keeps the model small (two bilinear forms + one linear gate, no MLP, no activation
function beyond the gate's own sigmoid) -- same "keep the scorer simple" choice this project's
other gates make, just swapped for a softmax-over-a-set loss instead of independent binary
classification.

Unlike gate v2/v3 (which never look at the raw BGE vectors, only at cosine folded into
emb_score outside the gate, and which train ONE model per pooled transition j), this model:
  - takes the raw BGE query/passage embeddings as PCA'd features instead of a pre-computed
    cosine scalar, so it can learn how to compare them rather than being handed one
    hand-picked summary statistic
  - is ONE model shared across all hop positions (no per-j split) -- h_after already encodes
    "what happened at every earlier hop" by construction (it's the hidden state after the
    whole accumulated prefix), so hop position doesn't need its own separate model

PCA is fit ONCE (from hop 1's collected data, see train_online_wavefront.py) and never
refit -- only .transform() is called for hop 2/3/4's data, so the feature space the scorer
sees stays fixed while training continues incrementally across hops.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePCA:
    """Four (or five, see include_h_prev) independent PCA transforms, fit once on hop-1 data,
    reused (transform only) after."""

    def __init__(self, pca_dim: int = 64, seed: int = 0, include_h_prev: bool = False):
        from sklearn.decomposition import PCA

        self.pca_dim = pca_dim
        self.seed = seed
        self.include_h_prev = include_h_prev
        self._PCA = PCA
        self.q_pca: Any = None
        self.d_pca: Any = None
        self.h_pca: Any = None
        self.delta_pca: Any = None
        # Ablation only (see MLPGateModel docstring): h_after - h_prev already tells the model
        # "how much did this specific candidate change the state", but not "what was the state
        # right before this pick" on its own -- h_prev as its OWN branch gives the model that
        # directly, on top of h_after/delta, instead of only the single local difference.
        self.hprev_pca: Any = None
        self.fitted = False

    def fit(self, q_vecs: np.ndarray, d_vecs: np.ndarray, h_vecs: np.ndarray, delta_vecs: np.ndarray,
            hprev_vecs: np.ndarray | None = None) -> None:
        def _fit_one(X: np.ndarray):
            nc = min(self.pca_dim, X.shape[0] - 1, X.shape[1])
            pca = self._PCA(n_components=nc, random_state=self.seed)
            pca.fit(X.astype(np.float64))
            return pca

        self.q_pca = _fit_one(q_vecs)
        self.d_pca = _fit_one(d_vecs)
        self.h_pca = _fit_one(h_vecs)
        self.delta_pca = _fit_one(delta_vecs)
        if self.include_h_prev:
            if hprev_vecs is None:
                raise ValueError("include_h_prev=True but no hprev_vecs given to fit()")
            self.hprev_pca = _fit_one(hprev_vecs)
        self.fitted = True

    def transform(self, q_vecs: np.ndarray, d_vecs: np.ndarray, h_vecs: np.ndarray, delta_vecs: np.ndarray,
                  hprev_vecs: np.ndarray | None = None) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeaturePCA.fit() must be called (on hop-1 data) before transform()")
        parts = [
            self.q_pca.transform(q_vecs.astype(np.float64)),
            self.d_pca.transform(d_vecs.astype(np.float64)),
            self.h_pca.transform(h_vecs.astype(np.float64)),
            self.delta_pca.transform(delta_vecs.astype(np.float64)),
        ]
        if self.include_h_prev:
            if hprev_vecs is None:
                raise ValueError("include_h_prev=True but no hprev_vecs given to transform()")
            parts.append(self.hprev_pca.transform(hprev_vecs.astype(np.float64)))
        return np.concatenate(parts, axis=1)

    @property
    def out_dim(self) -> int:
        pcas = [self.q_pca, self.d_pca, self.h_pca, self.delta_pca]
        if self.include_h_prev:
            pcas.append(self.hprev_pca)
        return sum(p.n_components_ for p in pcas)

    @property
    def branch_dims(self) -> tuple[int, ...]:
        """(q_dim, d_dim, h_dim, delta_dim[, hprev_dim]) -- how transform()'s output columns
        split up. ListwiseGateModel only accepts the first four (its bilinear pairing has no
        slot for a 5th branch) -- include_h_prev is for MLPGateModel, which is dim-agnostic."""
        dims = [
            self.q_pca.n_components_, self.d_pca.n_components_,
            self.h_pca.n_components_, self.delta_pca.n_components_,
        ]
        if self.include_h_prev:
            dims.append(self.hprev_pca.n_components_)
        return tuple(dims)


class ListwiseGateModel(nn.Module):
    """score = g*score_qd + (1-g)*score_hd, score_qd=q^T W_qd d, score_hd=h^T W_hd delta,
    g=sigmoid(linear([q,d,h,delta])) -- see module docstring for the reasoning behind each
    piece. forward() still takes a single concatenated feature tensor (same
    [q_dim+d_dim+h_dim+delta_dim] layout FeaturePCA.transform produces) and splits it
    internally, so callers don't need to change how they build inputs."""

    def __init__(self, q_dim: int, d_dim: int, h_dim: int, delta_dim: int):
        super().__init__()
        self.q_dim, self.d_dim, self.h_dim, self.delta_dim = q_dim, d_dim, h_dim, delta_dim
        self.bilinear_qd = nn.Bilinear(q_dim, d_dim, 1, bias=False)
        self.bilinear_hd = nn.Bilinear(h_dim, delta_dim, 1, bias=False)
        self.gate = nn.Linear(q_dim + d_dim + h_dim + delta_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, d, h, delta = torch.split(x, [self.q_dim, self.d_dim, self.h_dim, self.delta_dim], dim=-1)
        score_qd = self.bilinear_qd(q, d).squeeze(-1)
        score_hd = self.bilinear_hd(h, delta).squeeze(-1)
        g = torch.sigmoid(self.gate(x).squeeze(-1))
        return g * score_qd + (1.0 - g) * score_hd

    @property
    def in_dim(self) -> int:
        return self.q_dim + self.d_dim + self.h_dim + self.delta_dim


class MLPGateModel(nn.Module):
    """Ablation, not the production design: a plain MLP with a nonlinear activation (ReLU) over
    the FULL concatenated feature vector -- the same 4 branches (q,d,h,delta) ListwiseGateModel
    gets, optionally plus a 5th (h_prev, see FeaturePCA.include_h_prev) for an explicit "what was
    the state right before this pick" signal on top of h_after/delta alone. Two hidden layers.

    Exists to answer one specific question empirically: is the bilinear+gate model's small,
    deliberately-restricted capacity (see ListwiseGateModel's docstring -- "no MLP, no activation
    function beyond the gate's own sigmoid" was a design choice, not an oversight) actually
    leaving accuracy on the table on this mixed-source data, or is the PCA-dim bottleneck (see
    train_online_wavefront.py's --pca-dim ablation) the whole story? Run side by side with
    ListwiseGateModel on identical features via --model-type mlp; NOT a proposed replacement for
    the production gate."""

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def listwise_loss(model: nn.Module, groups: list[tuple[torch.Tensor, int]]) -> torch.Tensor:
    """
    groups: list of (candidate_features [n_i, in_dim], gold_index) for each hop instance
    that actually has gold in its candidate pool (hops without gold in the pool are dropped
    upstream -- there is no valid softmax target for them).
    """
    losses = []
    for feats, gold_idx in groups:
        logits = model(feats)
        losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([gold_idx], device=feats.device)))
    return torch.stack(losses).mean()


def groups_accuracy(model: nn.Module, groups: list[tuple[torch.Tensor, int]]) -> float:
    """Fraction of groups where argmax(model(feats)) == gold_idx -- the same "did it pick the
    gold candidate" signal pick_accuracy reports, just usable as a validation metric. Always
    no_grad (this is only ever used for monitoring, never for a training step)."""
    if not groups:
        return 0.0
    n_correct = 0
    with torch.no_grad():
        for feats, gold_idx in groups:
            pred = int(torch.argmax(model(feats)).item())
            if pred == gold_idx:
                n_correct += 1
    return n_correct / len(groups)


def save_artifact(path, feature_pca: FeaturePCA, model: nn.Module, meta: dict) -> None:
    import joblib

    is_mlp = isinstance(model, MLPGateModel)
    payload = {
        "q_pca": feature_pca.q_pca, "d_pca": feature_pca.d_pca,
        "h_pca": feature_pca.h_pca, "delta_pca": feature_pca.delta_pca,
        "hprev_pca": feature_pca.hprev_pca if feature_pca.include_h_prev else None,
        "pca_dim": feature_pca.pca_dim,
        "model_type": "mlp" if is_mlp else "bilinear",
        "model_state_dict": model.state_dict(),
        "meta": meta,
    }
    if is_mlp:
        payload.update({"in_dim": model.in_dim, "hidden_dim": model.hidden_dim})
    else:
        payload.update({"q_dim": model.q_dim, "d_dim": model.d_dim,
                         "h_dim": model.h_dim, "delta_dim": model.delta_dim})
    joblib.dump(payload, path)


def load_artifact(path) -> tuple[FeaturePCA, nn.Module, dict]:
    import joblib

    obj = joblib.load(path)
    include_h_prev = obj.get("hprev_pca") is not None
    fp = FeaturePCA(pca_dim=obj["pca_dim"], include_h_prev=include_h_prev)
    fp.q_pca, fp.d_pca, fp.h_pca, fp.delta_pca = obj["q_pca"], obj["d_pca"], obj["h_pca"], obj["delta_pca"]
    if include_h_prev:
        fp.hprev_pca = obj["hprev_pca"]
    fp.fitted = True

    model_type = obj.get("model_type", "bilinear")  # old artifacts (pre-MLP-ablation) had no such key
    if model_type == "mlp":
        model: nn.Module = MLPGateModel(obj["in_dim"], hidden_dim=obj.get("hidden_dim", 128))
    else:
        model = ListwiseGateModel(obj["q_dim"], obj["d_dim"], obj["h_dim"], obj["delta_dim"])
    model.load_state_dict(obj["model_state_dict"])
    model.eval()
    return fp, model, obj["meta"]
