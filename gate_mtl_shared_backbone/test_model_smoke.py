#!/usr/bin/env python3
"""Smoke test for MTLGateModel -- no real data yet, just proves the architecture's shapes,
forward/backward pass, and masking are correct before wiring in traces_v2/hidden_states_v2."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MTLGateModel, mtl_loss  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    B, K_max, D = 6, 4, 8192  # batch of 6 traces, max 4 hops, concat(h_after,delta)=8192-d
    model = MTLGateModel(input_dim=D)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    # fake a batch: variable K per trace (2,3,4,2,3,4), padded to K_max, mask marks real hops
    Ks = [2, 3, 4, 2, 3, 4]
    x = torch.randn(B, K_max, D)
    mask = torch.zeros(B, K_max)
    evidence_label = torch.zeros(B, K_max)
    hop_answer_label = torch.zeros(B, K_max)
    final_f1_label = torch.zeros(B, K_max)
    for i, K in enumerate(Ks):
        mask[i, :K] = 1.0
        evidence_label[i, :K] = torch.randint(0, 2, (K,)).float()
        hop_answer_label[i, :K] = torch.randint(0, 2, (K,)).float()
        f1 = torch.rand(1).item()  # ONE f1 per trace, broadcast to all its real hops
        final_f1_label[i, :K] = f1

    out = model(x)
    for k, v in out.items():
        assert v.shape == (B, K_max), f"{k} shape {v.shape} != {(B, K_max)}"
    print("forward output shapes ok:", {k: tuple(v.shape) for k, v in out.items()})

    # padding sanity: logits at padded positions should NOT affect loss (mask=0 there)
    losses = mtl_loss(
        out, evidence_label=evidence_label, hop_answer_label=hop_answer_label,
        final_f1_label=final_f1_label, mask=mask,
    )
    print("losses:", {k: round(v.item(), 4) for k, v in losses.items()})

    losses["loss_total"].backward()
    grad_norms = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
    n_with_grad = sum(1 for g in grad_norms.values() if g > 0)
    n_total = len(list(model.parameters()))
    print(f"params with nonzero grad after backward: {n_with_grad}/{n_total}")
    assert n_with_grad == n_total, "some parameters got no gradient -- a head or the trunk is disconnected"

    # verify masking actually matters: change a PADDED position's input drastically, loss should not move.
    # must eval() first -- dropout is stochastic per forward() call in train mode, which would make
    # out vs out2 differ for a reason that has nothing to do with padding leakage.
    model.eval()
    with torch.no_grad():
        out = model(x)
        losses = mtl_loss(
            out, evidence_label=evidence_label, hop_answer_label=hop_answer_label,
            final_f1_label=final_f1_label, mask=mask,
        )
    x2 = x.clone()
    x2[0, Ks[0]:, :] += 100.0  # corrupt only padding region of trace 0
    with torch.no_grad():
        out2 = model(x2)
    losses2 = mtl_loss(
        out2, evidence_label=evidence_label, hop_answer_label=hop_answer_label,
        final_f1_label=final_f1_label, mask=mask,
    )
    diff = abs(losses["loss_total"].item() - losses2["loss_total"].item())
    print(f"loss_total unchanged after corrupting only padded input: diff={diff:.8f}")
    assert diff < 1e-6, "masking is leaking padded positions into the loss"

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
