#!/usr/bin/env python3
"""
Step 5 test (see update_doc/0907/0907update.md section 9): extract RAW (non-PCA) hidden states
for a couple of already-validated traces, save to npz, reload it back and sanity-check shapes +
that the pair's correct/wrong hidden states are actually different vectors (not accidentally
duplicated) -- confirms the hidden-state wiring before batching/scaling up.

Usage:
  CUDA_VISIBLE_DEVICES=1 python step05_hidden_states_test.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TRACES_DIR = PROJECT_ROOT / "traces"
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TRACES_DIR))
sys.path.insert(0, str(RETRIEVAL_DIR))

import numpy as np  # noqa: E402
from dataset_loaders import get_decompose_path, load_decompose_bundle, load_raw_index  # noqa: E402
from trace_evidence import gold_evidence_musique, gold_hop_answers_musique  # noqa: E402
from run_retrieval_exp_wavefront import (  # noqa: E402
    BatchVllmGenerator, load_gate_model, warmup_gate_model_memory,
)
from recipe2_core import run_recipe2_trace, add_final_answer, add_hidden_states  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=HERE / "scratch_hidden_states")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    target_ids = ["3hop1__257997_104557_161232", "4hop3__695568_769559_129926_718885"]

    decompose_path = get_decompose_path("musique", "train", mode="gt")
    decomp, raw_id_map = load_decompose_bundle(decompose_path, include_enhance=False)
    raw_index = load_raw_index("musique", "train")

    cases = []
    for did in target_ids:
        raw_id = raw_id_map.get(did, did)
        record = raw_index[raw_id]
        sub_qs = decomp[did]
        gold_texts, gold_idxs = gold_evidence_musique(record)
        hop_answers = gold_hop_answers_musique(record)
        k = min(len(sub_qs), len(gold_idxs), len(hop_answers))
        cases.append((did, record, sub_qs[:k], gold_idxs[:k], gold_texts[:k]))

    print("Loading gate model (HF, for hidden states) ...")
    gate_args = argparse.Namespace(
        model=args.model, dtype="bfloat16", gate_device="cuda:0", attn_implementation=None,
    )
    gate_model, gate_tokenizer = load_gate_model(gate_args)
    warmup_gate_model_memory(gate_model, gate_tokenizer, batch_size=8)

    print("Loading vLLM generator ...")
    gen = BatchVllmGenerator(
        args.model, dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for did, record, sub_qs, gold_idxs, gold_texts in cases:
        trace = run_recipe2_trace(
            case_id=did, question=record.get("question", ""), sub_qs=sub_qs,
            gold_idxs=gold_idxs, gold_texts=gold_texts, paragraphs=record.get("paragraphs") or [],
            generator=gen,
        )
        add_final_answer(trace, record, gen)
        npz_path = args.out_dir / f"{trace['trace_id']}.npz"
        add_hidden_states(trace, gate_model=gate_model, gate_tokenizer=gate_tokenizer,
                           layer=args.layer, npz_path=npz_path)

        print(f"\n=== {trace['trace_id']}  K={trace['K']}  n_pairs={len(trace['pairs'])} ===")
        print(f"  npz saved: {npz_path}")

        z = np.load(npz_path, allow_pickle=True)
        hidden = z["hidden"]
        print(f"  hidden.shape = {hidden.shape}  (expect ({trace['K'] + 1}, 4096))")
        print(f"  dtype = {hidden.dtype}")
        # h_0 should differ from h_1 (question-only vs question+hop1) -- sanity check they're
        # not accidentally all identical/zero.
        diffs = [float(np.linalg.norm(hidden[i] - hidden[i - 1])) for i in range(1, hidden.shape[0])]
        print(f"  ||h_i - h_{{i-1}}|| per hop: {[round(d, 3) for d in diffs]}")

        if trace["pairs"]:
            pc = z["pair_hidden_correct"]
            pw = z["pair_hidden_wrong"]
            print(f"  pair_hidden_correct.shape={pc.shape}  pair_hidden_wrong.shape={pw.shape}")
            for i, pid in enumerate(z["pair_ids"]):
                dist = float(np.linalg.norm(pc[i] - pw[i]))
                hop = trace["pairs"][i]["hop"]
                h_prev_norm = float(np.linalg.norm(hidden[hop - 1]))
                print(f"  pair[{i}] id={pid}  ||h_after_correct - h_after_wrong|| = {dist:.3f}  "
                      f"(non-zero means the two variants are genuinely different vectors, not "
                      f"duplicated; ||h_prev||={h_prev_norm:.3f} for scale reference)")


if __name__ == "__main__":
    main()
