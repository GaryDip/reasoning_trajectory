#!/usr/bin/env python3
"""
Re-run ONLY the final-answer generation step against an existing wavefront run's logged
cases, instead of a full end-to-end rerun (retrieval + gate + per-hop short answers all
stay exactly as already computed) -- much cheaper way to A/B test a new final-reader prompt
(see build_final_reader_cot_prompt_comparison_reasoning in run_retrieval_exp_wavefront.py).

The case JSONL from a rawprefix run already has everything needed except the per-hop
EXPANDED subquestion text, which is reconstructed deterministically:
  raw_sq       = decompose_idx[id][j]                       (from the BART decompose file)
  expanded_q   = expand_hop_template(raw_sq, prior[:j])      (same function production uses)
  evidence     = paragraphs_by_id[id][para_ids[j]]["paragraph_text"]

Usage:
  python replay_final_reader.py \
    --dataset 2wiki \
    --cases-file results/20260813_072509_gate_v3_rawprefix_2wiki_comparisonhint_full/retrieval_cases_dev_gate_v3_rawprefix_2wiki_comparisonhint_full.jsonl \
    --final-reader-prompt comparison_reasoning \
    --run-tag gate_v3_2wiki_comparisonreasoning_replay
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_retrieval_exp import (  # noqa: E402
    HOTPOT_DEV_FILE,
    MUSIQUE_DIR,
    TWOWIKI_DEV_FILE,
    AnswerAccum,
    _get_pipeline_helpers,
    load_dataset_records,
    load_decompose_index,
    resolve_decompose_file,
    sanitize_run_tag,
)
from run_retrieval_exp_wavefront import (  # noqa: E402
    FINAL_READER_SYSTEM_PROMPT,
    BatchVllmGenerator,
    build_final_reader_cot_prompt,
    build_final_reader_cot_prompt_comparison_hint,
    build_final_reader_cot_prompt_comparison_positive,
    build_final_reader_cot_prompt_comparison_reasoning,
    parse_final_answer_from_cot,
)

FINAL_READER_PROMPT_BUILDERS = {
    "default": build_final_reader_cot_prompt,
    "comparison_hint": build_final_reader_cot_prompt_comparison_hint,
    "comparison_reasoning": build_final_reader_cot_prompt_comparison_reasoning,
    "comparison_positive": build_final_reader_cot_prompt_comparison_positive,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique", "2wiki", "hotpot"], default="musique")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--twowiki-file", type=Path, default=TWOWIKI_DEV_FILE)
    ap.add_argument("--hotpot-file", type=Path, default=HOTPOT_DEV_FILE)
    ap.add_argument("--decompose-mode", choices=["gt", "bart_decompose"], default="bart_decompose")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--cases-file", type=Path, required=True,
                     help="retrieval_cases_*.jsonl from an already-completed rawprefix run.")
    ap.add_argument("--final-reader-prompt", choices=list(FINAL_READER_PROMPT_BUILDERS), default="comparison_reasoning")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--results-root", type=Path, default=HERE / "results")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens-final", type=int, default=256,
                     help="Higher than the usual 128 default -- comparison_reasoning asks "
                          "for an explicit reasoning line before the final answer, which "
                          "needs more headroom than a short-span-only answer.")
    return ap.parse_args()


def main() -> None:
    run_t0 = time.perf_counter()
    args = parse_args()
    args.decompose_file = resolve_decompose_file(
        args.decompose_mode, args.split, args.decompose_file, args.dataset
    )

    ds_args = argparse.Namespace(
        dataset=args.dataset, musique_dir=args.musique_dir, split=args.split,
        twowiki_file=args.twowiki_file, hotpot_file=args.hotpot_file,
    )
    records = load_dataset_records(ds_args)
    decompose_idx = load_decompose_index(args.decompose_file)
    (expand_hop_template, *_rest, judge_answer_official) = _get_pipeline_helpers()

    cases = [json.loads(line) for line in args.cases_file.open(encoding="utf-8") if line.strip()]
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} cases from {args.cases_file}")

    tag = args.run_tag or f"replay_final_reader_{args.dataset}_{args.final_reader_prompt}"
    out_dir = args.out_dir or (args.results_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{sanitize_run_tag(tag)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cases = out_dir / "retrieval_cases_dev.jsonl"
    out_metrics = out_dir / "retrieval_exp_dev.json"
    print(f"Output directory: {out_dir}")

    reader_prompt_fn = FINAL_READER_PROMPT_BUILDERS[args.final_reader_prompt]

    skipped = 0
    reconstructed: list[dict[str, Any]] = []
    for case in cases:
        eid = str(case["id"])
        row = records.get(eid)
        raw_sqs = decompose_idx.get(eid)
        para_ids = case.get("para_ids") or []
        prior = case.get("prior") or []
        if row is None or not raw_sqs or len(para_ids) != len(prior) or not prior:
            skipped += 1
            continue
        para_by_idx = {int(p["idx"]): p for p in row.get("paragraphs") or []}
        hop_steps: list[tuple[str, str]] = []
        ok = True
        for j in range(len(prior)):
            if j >= len(raw_sqs):
                ok = False
                break
            expanded_q = expand_hop_template(raw_sqs[j], prior[:j])
            para = para_by_idx.get(int(para_ids[j]))
            if para is None:
                ok = False
                break
            hop_steps.append((expanded_q, (para.get("paragraph_text") or "").strip()))
        if not ok:
            skipped += 1
            continue
        reconstructed.append({
            "id": eid, "row": row, "question": case["question"], "gold_answer": case.get("gold_answer"),
            "hop_steps": hop_steps, "prior": prior,
            "logged_final_answer": case.get("final_answer"), "logged_em": case.get("em"), "logged_f1": case.get("f1"),
        })
    print(f"Reconstructed {len(reconstructed)} cases ({skipped} skipped: missing row/decompose/paragraph)")

    print("Loading vLLM generator ...", flush=True)
    generator = BatchVllmGenerator(
        args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
    )

    prompts = [reader_prompt_fn(c["question"], c["hop_steps"], c["prior"]) for c in reconstructed]
    raw_outputs = generator.generate_chat_batch(
        FINAL_READER_SYSTEM_PROMPT, prompts, args.max_new_tokens_final, desc="final-reader-replay",
    )

    answer_accum = AnswerAccum()
    n_flipped_to_correct = 0
    n_flipped_to_wrong = 0
    with out_cases.open("w", encoding="utf-8") as f:
        for c, raw in zip(reconstructed, raw_outputs):
            fallback = c["prior"][-1] if c["prior"] else ""
            final_answer = parse_final_answer_from_cot(raw, fallback=fallback)
            em, f1 = judge_answer_official(final_answer, c["row"])
            answer_accum.update(em, f1)
            if em == 1 and not c["logged_em"]:
                n_flipped_to_correct += 1
            elif not em and c["logged_em"]:
                n_flipped_to_wrong += 1
            f.write(json.dumps({
                "id": c["id"], "question": c["question"], "gold_answer": c["gold_answer"],
                "final_answer": final_answer, "em": em, "f1": f1,
                "logged_final_answer": c["logged_final_answer"], "logged_em": c["logged_em"], "logged_f1": c["logged_f1"],
                "raw_output": raw,
            }, ensure_ascii=False) + "\n")

    run_wall_sec = time.perf_counter() - run_t0
    results = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cases_file": str(args.cases_file), "n_cases": len(reconstructed), "n_skipped": skipped,
        "timing": {"wall_clock_sec": round(run_wall_sec, 2)},
        "answer_overall": answer_accum.result(),
        "n_flipped_wrong_to_correct": n_flipped_to_correct,
        "n_flipped_correct_to_wrong": n_flipped_to_wrong,
    }
    out_metrics.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMetrics saved to {out_metrics}")
    print(f"answer_overall: {results['answer_overall']}")
    print(f"flipped wrong->correct: {n_flipped_to_correct}  flipped correct->wrong: {n_flipped_to_wrong}")


if __name__ == "__main__":
    main()
