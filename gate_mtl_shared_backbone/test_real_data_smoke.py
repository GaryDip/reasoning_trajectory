#!/usr/bin/env python3
"""Real-data end-to-end smoke test: traces_v2 + hidden_states_v2 -> MTLGateDataset -> DataLoader
-> MTLGateModel -> loss -> a handful of optimizer steps, to confirm the whole path actually runs
and the loss moves (not just synthetic-tensor shape checks like test_model_smoke.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MTLGateModel, mtl_loss  # noqa: E402
from dataset import MTLGateDataset, collate_pad  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    torch.set_num_threads(4)  # avoid oversubscribing this shared machine's CPUs
    traces_path = REPO / "new_data_recipe/output_full/traces_v2/musique/train_with_hop_labels.jsonl"
    hs_dir = REPO / "new_data_recipe/output_full/hidden_states_v2/musique/train"

    ds = MTLGateDataset(traces_path, hs_dir, limit=500)
    print(f"[{len(ds)} traces] preloading all items into memory once (avoids re-hitting disk "
          f"for every one of 30 epochs) ...", flush=True)
    items = [ds[i] for i in range(len(ds))]
    print("preload done", flush=True)

    dl = DataLoader(items, batch_size=16, shuffle=True, collate_fn=collate_pad)

    batch = next(iter(dl))
    print("batch x:", tuple(batch["x"].shape), "mask:", tuple(batch["mask"].shape), flush=True)
    print("sample K per trace in this batch:", batch["mask"].sum(dim=1).tolist(), flush=True)
    print("evidence_label mean (this batch):", batch["evidence_label"][batch["mask"] > 0].mean().item(), flush=True)
    print("hop_answer_label mean (this batch):", batch["hop_answer_label"][batch["mask"] > 0].mean().item(), flush=True)
    print("final_f1_label mean (this batch):", batch["final_f1_label"][batch["mask"] > 0].mean().item(), flush=True)

    D = batch["x"].shape[-1]
    model = MTLGateModel(input_dim=D)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n--- training a handful of steps on 500 real traces (in-memory) to confirm loss moves ---", flush=True)
    model.train()
    losses_over_time = []
    for step in range(30):
        total_loss_acc = 0.0
        n_batches = 0
        for batch in dl:
            out = model(batch["x"])
            losses = mtl_loss(
                out, evidence_label=batch["evidence_label"], hop_answer_label=batch["hop_answer_label"],
                final_f1_label=batch["final_f1_label"], mask=batch["mask"],
            )
            opt.zero_grad()
            losses["loss_total"].backward()
            opt.step()
            total_loss_acc += losses["loss_total"].item()
            n_batches += 1
        avg = total_loss_acc / n_batches
        losses_over_time.append(avg)
        if step % 5 == 0 or step == 29:
            print(f"  epoch {step:2d}: avg loss_total = {avg:.4f}  "
                  f"(evidence={losses['loss_evidence'].item():.4f} "
                  f"hop_answer={losses['loss_hop_answer'].item():.4f} "
                  f"final_f1={losses['loss_final_f1'].item():.4f})", flush=True)

    print(f"\nloss_total: epoch0={losses_over_time[0]:.4f} -> epoch29={losses_over_time[-1]:.4f}")
    assert losses_over_time[-1] < losses_over_time[0], "loss did not go down at all over 30 epochs on 500 traces"
    print("SMOKE TEST PASSED (loss decreased on real data)")


if __name__ == "__main__":
    main()
