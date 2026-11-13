#!/usr/bin/env python3
"""Prove that `truncate_gate_model_to_layer` changes nothing the gate reads.

Runs on CPU with a tiny randomly-initialised Llama, so it needs no checkpoint and no GPU, and
compares the full model against the truncated one on the exact quantity the gate consumes:
`hidden_states[layer + 1]` at the last non-padding position, for a padded batch.

    python retrieval/verify_gate_truncation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_retrieval_exp_wavefront import truncate_gate_model_to_layer  # noqa: E402

N_LAYERS, LAYER = 8, 3          # read hidden_states[4]: the raw output of block 3


def read_gate_vector(model, input_ids, attention_mask, layer: int) -> torch.Tensor:
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[layer + 1]
    last = attention_mask.size(1) - 1 - torch.flip(attention_mask, dims=[1]).argmax(dim=1)
    return hs[torch.arange(input_ids.size(0)), last].float()


def main() -> int:
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=N_LAYERS, num_attention_heads=4,
                      num_key_value_heads=2, torch_dtype=torch.float32)
    model = LlamaForCausalLM(cfg).eval()

    # a padded batch, like the real gate batches (right padding, variable lengths)
    input_ids = torch.randint(0, 256, (3, 12))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 9:] = 0
    attention_mask[2, 5:] = 0
    input_ids = input_ids * attention_mask

    before = read_gate_vector(model, input_ids, attention_mask, LAYER)
    kept = truncate_gate_model_to_layer(model, LAYER)
    after = read_gate_vector(model, input_ids, attention_mask, LAYER)

    same = torch.equal(before, after)
    max_abs = (before - after).abs().max().item()
    print(f"blocks: {N_LAYERS} -> {kept}   (gate reads hidden_states[{LAYER + 1}])")
    print(f"按位相同: {same}   最大绝对差: {max_abs:.3e}")

    # and the control: truncating one block too early MUST change the value, otherwise the test
    # is not actually sensitive to the truncation point
    torch.manual_seed(0)
    model2 = LlamaForCausalLM(cfg).eval()
    ref = read_gate_vector(model2, input_ids, attention_mask, LAYER)
    truncate_gate_model_to_layer(model2, LAYER - 1)
    with torch.inference_mode():
        out = model2(input_ids=input_ids, attention_mask=attention_mask,
                     output_hidden_states=True, use_cache=False)
    too_short = len(out.hidden_states) <= LAYER + 1
    print(f"\n对照：截到第 {LAYER} 块时 hidden_states 长度 {len(out.hidden_states)}，"
          f"取不到 [{LAYER + 1}]: {too_short}")

    if not same:
        print("失败：截断改变了 gate 读取的向量")
        return 1
    print("通过：截断是精确优化")
    return 0


if __name__ == "__main__":
    sys.exit(main())
