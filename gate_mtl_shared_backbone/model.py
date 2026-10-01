#!/usr/bin/env python3
"""
Shared-backbone multi-task gate: one small MLP trunk applied per-hop, three per-hop heads on
top of it (see update_doc discussion 2026-09-04):

  task1 (head_evidence):    is this hop's COMMITTED EVIDENCE correct (== gold)?      label: hop's is_correct
  task2 (head_hop_answer):  is this hop's GENERATED SHORT ANSWER correct?            label: hop_answer_em/f1 (needs backfill, not yet in traces_v2)
  task3 (head_final_f1):    "value function" -- given the trace so far, how likely    label: trace's final_answer_f1,
                            is the FINAL answer to come out correct?                         broadcast to every hop

All three heads are per-hop (no pooling/sequence model) -- task3's label is the same single
number for every hop of a given trace (the trace only has one final answer), which is exactly
the point: early hops see less of the trajectory and should be less certain, later hops more so.

Input feature per hop: concat(h_after, delta_h) -- RAW (non-PCA) last-token hidden states from
hidden_states_v2/*.npz, hidden_dim=4096 each (Llama-3.1-8B-Instruct, layer 23 by default) ->
8192-dim per hop. h_after is the cumulative-prefix hidden state AFTER this hop's evidence is
appended (prefix_idx = hop number); delta_h = h_after - h_prev (prefix_idx - 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MTLGateModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 8192,     # concat(h_after[4096], delta_h[4096])
        trunk_hidden: int = 512,
        trunk_out: int = 128,
        head_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.trunk_out = trunk_out
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, trunk_hidden),
            nn.LayerNorm(trunk_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_out),
            nn.LayerNorm(trunk_out),
            nn.GELU(),
        )

        def make_head() -> nn.Module:
            return nn.Sequential(
                nn.Linear(trunk_out, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 1),
            )

        self.head_evidence = make_head()    # task 1
        self.head_hop_answer = make_head()  # task 2
        self.head_final_f1 = make_head()    # task 3

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [B, K, input_dim] (per-hop features, padded to the batch's max K).
        Returns per-hop RAW logits (no sigmoid) for all three heads, each [B, K] -- apply
        sigmoid / BCEWithLogitsLoss outside, and mask out padding positions with `mtl_loss`."""
        z = self.backbone(x)  # [B, K, trunk_out]
        return {
            "evidence_logit": self.head_evidence(z).squeeze(-1),
            "hop_answer_logit": self.head_hop_answer(z).squeeze(-1),
            "final_f1_logit": self.head_final_f1(z).squeeze(-1),
        }


def mtl_loss(
    outputs: dict[str, torch.Tensor],
    *,
    evidence_label: torch.Tensor,    # [B, K] in {0,1}
    hop_answer_label: torch.Tensor,  # [B, K] in {0,1}
    final_f1_label: torch.Tensor,    # [B, K] in [0,1] (soft target, same value repeated across a trace's K)
    mask: torch.Tensor,              # [B, K] 1.0 for real hops, 0.0 for padding
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, torch.Tensor]:
    """Masked BCE-with-logits per task (task3 uses a soft [0,1] target -- BCE supports that
    directly, no need to binarize final_answer_f1). Returns per-task losses + the weighted sum
    so callers can log task-level curves separately."""
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def masked_mean(loss: torch.Tensor) -> torch.Tensor:
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    l1 = masked_mean(bce(outputs["evidence_logit"], evidence_label.float()))
    l2 = masked_mean(bce(outputs["hop_answer_logit"], hop_answer_label.float()))
    l3 = masked_mean(bce(outputs["final_f1_logit"], final_f1_label.float()))
    w1, w2, w3 = weights
    total = w1 * l1 + w2 * l2 + w3 * l3
    return {"loss_evidence": l1, "loss_hop_answer": l2, "loss_final_f1": l3, "loss_total": total}
