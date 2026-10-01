"""GateModel: predicts ONE mixing ratio (gate in [0,1]) per hop (shared across every
candidate in that hop's pool), not a per-candidate score, and not an open-ended lambda
coefficient either. The rerank formula is a bounded CONVEX combination of the two
already-frozen, per-candidate signals:

    final_score_i = gate * zscore(emb_score)_i + (1 - gate) * zscore(-abnormal_score)_i

emb_score (BGE) and abnormal_score (error_propagation_probe) are both frozen; this model
only learns how much to trust one versus the other, conditioned on hop-level context. The
z-scoring (done per-pool, at loss/inference time — see train_lambda_model.py::compute_loss
and wavefront_adaptive_lambda.py) puts both signals on a comparable scale first, so gate=0.5
genuinely means "trust both equally" instead of being at the mercy of whichever raw score
happens to have the larger numeric range.

Earlier version learned an unbounded `lambda` plugged into `emb_score - lambda*abnormal_score`
directly. Two failure modes came from that: (1) an unbounded softplus activation let lambda
run away to ~10 chasing lower training loss (abnormal_score then completely swamps emb_score,
generalizes badly); (2) capping it with lambda_max*sigmoid just moved the failure — training
collapsed to lambda == lambda_max for every hop (std 0.0), i.e. a degenerate constant, still
worse than production's own tuned lambda=0.25. A bounded gate over CALIBRATED signals removes
the structural incentive to run to an extreme: neither signal can be weighted "infinitely
more" than the other since the weights must sum to 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BGE_DIM = 768
LLAMA_DIM = 4096


class LambdaModel(nn.Module):
    """Name kept as LambdaModel for import-compatibility with the rest of this folder;
    it now predicts a gate in [0,1], not an open-ended lambda — see module docstring."""

    def __init__(self, hidden: int = 32, use_diff: bool = True):
        super().__init__()
        self.use_diff = use_diff
        self.proj_h = nn.Linear(LLAMA_DIM, BGE_DIM)
        in_dim = BGE_DIM * (4 if use_diff else 3)  # q_main, proj(h_prev), expanded_q, [diff]
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def forward(
        self, q_main_emb: torch.Tensor, h_prev: torch.Tensor, expanded_q_emb: torch.Tensor,
    ) -> torch.Tensor:
        """All three inputs are [B, dim] (dim=768 for the BGE ones, 4096 for h_prev).
        Returns gate, shape [B], in (0, 1) via plain sigmoid.

        h_prev is L2-normalized before the projection: raw Llama hidden states run ~150x
        larger in norm than the BGE embeddings (measured on real data: h_prev row-norm
        mean ~149 vs 1.0 for q_main_emb/expanded_q_emb, which SentenceTransformer already
        normalizes). Left unnormalized, that scale mismatch alone can push raw pre-
        activation values to extremes regardless of what the final combination step does."""
        h_prev = F.normalize(h_prev, dim=-1)
        h_proj = self.proj_h(h_prev)
        feats = [q_main_emb, h_proj, expanded_q_emb]
        if self.use_diff:
            feats.append(expanded_q_emb - q_main_emb)
        x = torch.cat(feats, dim=-1)
        raw = self.mlp(x).squeeze(-1)
        return torch.sigmoid(raw)
