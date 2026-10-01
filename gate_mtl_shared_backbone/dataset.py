#!/usr/bin/env python3
"""
traces_v2 + hidden_states_v2 -> per-trace [K, 8192] feature tensors + 3 label tensors, for
MTLGateModel. Feature per hop j (1-indexed): concat(h_after_j, delta_j) where h_after_j =
hidden[j] (cumulative-prefix hidden state after hop j's evidence) and delta_j = hidden[j] -
hidden[j-1], both read straight from hidden_states_v2/{trace_id}.npz's `hidden` array
(shape [K+1, hidden_dim], RAW/non-PCA). No PCA anywhere in this path.

Labels per hop: evidence_label = is_correct (binary); hop_answer_label = hop_answer_f1
(continuous [0,1], soft BCE target -- not binarized EM, a near-miss short answer is real
partial-credit signal, same soft-label treatment as task3) (needs
backfill_hop_answer_labels.py to have been run first); final_f1_label = the trace's single
final_answer_f1, broadcast to every hop position (task3's "value function" framing).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MTLGateDataset(Dataset):
    def __init__(self, traces_path: Path | str, hidden_states_dir: Path | str, limit: int | None = None) -> None:
        self.hidden_states_dir = Path(hidden_states_dir)
        self.records: list[dict] = []
        n_skipped_missing_npz = 0
        n_skipped_missing_labels = 0
        with open(traces_path, encoding="utf-8") as f:
            for line in f:
                if limit is not None and len(self.records) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                npz_path = self.hidden_states_dir / f"{t['trace_id']}.npz"
                if not npz_path.is_file():
                    n_skipped_missing_npz += 1
                    continue
                if any(h.get("hop_answer_f1") is None for h in t["hops"]):
                    n_skipped_missing_labels += 1
                    continue
                if t.get("final_answer_f1") is None:
                    n_skipped_missing_labels += 1
                    continue
                self.records.append(t)
        print(f"MTLGateDataset: {len(self.records)} traces usable "
              f"({n_skipped_missing_npz} missing npz, {n_skipped_missing_labels} missing labels, skipped)")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        t = self.records[idx]
        npz_path = self.hidden_states_dir / f"{t['trace_id']}.npz"
        with np.load(npz_path) as d:
            hidden = d["hidden"].astype(np.float32)  # [K+1, hidden_dim]

        K = len(t["hops"])
        h_after = hidden[1:K + 1]       # [K, hidden_dim]
        h_prev = hidden[0:K]            # [K, hidden_dim]
        delta = h_after - h_prev        # [K, hidden_dim]
        x = np.concatenate([h_after, delta], axis=-1)  # [K, 2*hidden_dim]

        evidence_label = np.array([1.0 if h["is_correct"] else 0.0 for h in t["hops"]], dtype=np.float32)
        hop_answer_label = np.array([float(h["hop_answer_f1"]) for h in t["hops"]], dtype=np.float32)
        final_f1 = float(t["final_answer_f1"])
        final_f1_label = np.full((K,), final_f1, dtype=np.float32)

        return {
            "x": torch.from_numpy(x),
            "evidence_label": torch.from_numpy(evidence_label),
            "hop_answer_label": torch.from_numpy(hop_answer_label),
            "final_f1_label": torch.from_numpy(final_f1_label),
            "K": K,
            "trace_id": t["trace_id"],
        }


def collate_pad(batch: list[dict]) -> dict:
    """Pad a list of variable-K trace dicts to the batch's max K, with a [B,K] mask."""
    K_max = max(item["K"] for item in batch)
    B = len(batch)
    D = batch[0]["x"].shape[-1]

    x = torch.zeros(B, K_max, D)
    mask = torch.zeros(B, K_max)
    evidence_label = torch.zeros(B, K_max)
    hop_answer_label = torch.zeros(B, K_max)
    final_f1_label = torch.zeros(B, K_max)
    trace_ids = []
    for i, item in enumerate(batch):
        K = item["K"]
        x[i, :K] = item["x"]
        mask[i, :K] = 1.0
        evidence_label[i, :K] = item["evidence_label"]
        hop_answer_label[i, :K] = item["hop_answer_label"]
        final_f1_label[i, :K] = item["final_f1_label"]
        trace_ids.append(item["trace_id"])

    return {
        "x": x, "mask": mask, "evidence_label": evidence_label,
        "hop_answer_label": hop_answer_label, "final_f1_label": final_f1_label,
        "trace_ids": trace_ids,
    }
