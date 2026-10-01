#!/usr/bin/env python3
"""
Reader-ceiling probe: force GOLD evidence at every hop (skip retrieval/gate
entirely) and see what EM/F1 the exact same reader (hop-answer generation +
final-answer generation) achieves. This directly tests the hypothesis in
update_doc/0713/0713update.md section 3/5 — that recent retrieval-side
improvements (gate v2's per-j layer mixing, evidence beam search) both moved
retrieval metrics without moving answer EM/F1, suggesting the READER may be
capping answer quality, not evidence selection.

Two decompose modes, two different (both valid) ways of picking "gold"
evidence per hop:

  --decompose-mode gt (default): sub_questions[hop_j-1] IS gold hop j by
  construction (gt sub_questions come from the same question_decomposition
  list at the same index), so gold_decomp[hop_j-1]'s evidence is looked up
  directly — no ambiguity, no extra LLM call needed.

  --decompose-mode bart_decompose: BART's predicted hop j is NOT guaranteed
  to correspond to gold hop j (different order/granularity), so directly
  indexing gold_decomp[hop_j-1] can inject evidence that answers a DIFFERENT
  question than the one actually being asked at that hop. Instead, this
  script pools ALL of the example's gold paragraphs together and has the
  model itself SELECT (reusing prompt_select_passage_with_context, the same
  selection prompt production's greedy hop loop uses) whichever one best
  answers the CURRENT hop's subquestion — removing it from the pool once
  picked, so each gold paragraph is used at most once across hops. This
  still only ever offers genuinely correct (gold) evidence, no distractors,
  so it's still an oracle-evidence probe; it just also tests whether the
  reader can correctly MATCH gold evidence to the right hop when hop order
  isn't given for free. If the pool runs dry (BART predicts more hops than
  there is gold evidence for), it falls back to top-1 cosine retrieval.

Two possible outcomes once this is run for real:
  - EM/F1 is still unremarkable -> the ceiling is in the reader itself
    (given perfect evidence, it still can't answer correctly); further
    retrieval-side work (decompose beam search, etc.) has limited headroom
    until the reader is addressed.
  - EM/F1 is clearly high (well above current bart_decompose-based methods)
    -> evidence/decompose quality still matters a lot; the specific
    interventions tried so far (layer mixing, beam search) just haven't
    found the right lever yet.

Evidence source per hop (gold_direct / gold_selected / fallback_top1) is
tracked and reported so a surprisingly low gold-coverage rate — or a
surprisingly bad selection accuracy under bart_decompose — doesn't go
unnoticed.

Reuses (imports, does not copy) retrieval/run_retrieval_exp.py and
retrieval/run_retrieval_exp_wavefront.py's data loading, retrieval, vLLM
generation, and ExampleState/MethodState/init_example_state machinery — no
gate model is loaded at all (nothing is being scored, gold is looked up
directly), so this is lighter-weight than beam_search_retrieval.py.

Usage (needs a real GPU + Llama checkpoint + vLLM):
  python oracle_evidence_reader.py --limit 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))

from run_retrieval_exp import (  # noqa: E402
    MUSIQUE_DIR,
    _get_pipeline_helpers,
    embed_retrieval,
    load_dataset_records,
    load_decompose_index,
    normalize_short_answer,
    prepare_sub_questions,
    resolve_decompose_file,
    select_example_ids,
)
from run_retrieval_exp_wavefront import (  # noqa: E402
    FINAL_READER_SYSTEM_PROMPT,
    BatchVllmGenerator,
    build_final_reader_cot_prompt,
    init_example_state,
    parse_final_answer_from_cot,
    prompt_select_passage_with_context,
    prompt_short_answer_with_context,
)

METHOD = "oracle_reader"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique"], default="musique")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--decompose-mode", choices=["gt", "bart_decompose"], default="gt",
                     help="gt is the only mode where forcing gold_decomp[hop_j-1]'s evidence is "
                          "actually valid: gt sub_questions come from the same question_decomposition "
                          "list at the same index, so hop j always lines up with gold hop j. With "
                          "bart_decompose, BART's predicted hop j is NOT guaranteed to correspond to "
                          "gold hop j (different order/granularity) — forcing gold_decomp[hop_j-1]'s "
                          "evidence in that case can inject evidence that answers a DIFFERENT question "
                          "than the one actually being asked at that hop, invalidating the whole probe. "
                          "Only override to bart_decompose if you specifically want to (incorrectly) "
                          "compare against bart-based runs and accept that risk.")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--answerable-only", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens-answer", type=int, default=64)
    ap.add_argument("--max-new-tokens-final", type=int, default=128)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.decompose_file = resolve_decompose_file(
        args.decompose_mode, args.split, args.decompose_file, args.dataset
    )

    ds_args = argparse.Namespace(dataset=args.dataset, musique_dir=args.musique_dir, split=args.split)
    records = load_dataset_records(ds_args)
    decompose_idx = load_decompose_index(args.decompose_file)
    ids = sorted(set(records) & set(decompose_idx))
    if args.answerable_only:
        ids = [eid for eid in ids if records[eid].get("answerable", True)]
    ids = select_example_ids(ids, decompose_idx, limit=args.limit, limit_per_k=0, seed=args.sample_seed)

    (expand_hop_template, _prompt_select_passage, parse_json_choice,
     _prompt_short_answer, _prompt_concat_answer, _parse_json_answer_choice,
     judge_answer_official) = _get_pipeline_helpers()

    print(f"Examples: {len(ids)}  decompose={args.decompose_file}  (method={METHOD})")

    examples = []
    for eid in ids:
        row = records[eid]
        sub_qs = prepare_sub_questions(decompose_idx.get(eid) or [], decompose_mode=args.decompose_mode)
        if not sub_qs:
            continue
        examples.append(init_example_state(eid, row, sub_qs, [METHOD], []))

    # bart_decompose only: pool of ALL gold paragraphs per example, consumed
    # (removed) as hops pick from it — see module docstring for why this is
    # needed instead of directly indexing gold_decomp[hop_j-1].
    gold_pools: dict[str, list[dict[str, Any]]] = {}
    if args.decompose_mode == "bart_decompose":
        for ex in examples:
            pool = []
            seen_idx = set()
            for gd in ex.gold_decomp:
                try:
                    pi = int(gd.get("paragraph_support_idx"))
                except (TypeError, ValueError):
                    continue
                if pi in seen_idx:
                    continue
                para = next((p for p in ex.paragraphs if int(p.get("idx", -1)) == pi), None)
                if para is not None:
                    seen_idx.add(pi)
                    pool.append(para)
            gold_pools[ex.eid] = pool

    print("Loading vLLM generator ...", flush=True)
    generator = BatchVllmGenerator(
        args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
    )

    max_hops = max((ex.K for ex in examples), default=0)
    n_gold_hops = 0
    n_fallback_hops = 0

    for hop_j in range(1, max_hops + 1):
        print(f"\n=== hop {hop_j}/{max_hops} ===", flush=True)

        # Step 1: for every example still active at this hop, work out how
        # it'll get its evidence this hop. gt: gold_decomp[hop_j-1] directly.
        # bart_decompose: needs a selection among the example's remaining
        # gold pool (batched below) unless the pool is empty or has exactly
        # one candidate (nothing to actually select). No gate, no beam —
        # single forced path per example either way.
        direct_hits: list[tuple[Any, str, dict[str, Any]]] = []  # (ex, expanded_q, chosen_para)
        needs_selection: list[tuple[Any, str, list[dict[str, Any]]]] = []  # (ex, expanded_q, pool)
        needs_fallback: list[tuple[Any, str]] = []  # (ex, expanded_q)

        for ex in examples:
            if hop_j > ex.K:
                continue
            ms = ex.methods[METHOD]
            raw_sq = ex.sub_questions[hop_j - 1]
            expanded_q = expand_hop_template(raw_sq, ms.prior)

            if args.decompose_mode == "gt":
                hop_row = ex.hop_results[hop_j - 1]
                gold_pi = hop_row.get("gold_para_idx")
                chosen_para = None
                if bool(hop_row.get("has_gold_hop")) and gold_pi is not None:
                    chosen_para = next(
                        (p for p in ex.paragraphs if int(p.get("idx", -1)) == int(gold_pi)), None,
                    )
                if chosen_para is not None:
                    direct_hits.append((ex, expanded_q, chosen_para))
                else:
                    needs_fallback.append((ex, expanded_q))
                continue

            # bart_decompose
            pool = gold_pools.get(ex.eid) or []
            if not pool:
                needs_fallback.append((ex, expanded_q))
            elif len(pool) == 1:
                direct_hits.append((ex, expanded_q, pool[0]))
                pool.pop(0)
            else:
                needs_selection.append((ex, expanded_q, list(pool)))

        # Batched selection among each example's remaining gold pool.
        selected: list[tuple[Any, str, dict[str, Any]]] = []
        if needs_selection:
            selection_prompts = [
                prompt_select_passage_with_context(ex.q_main, ex.methods[METHOD].hop_steps,
                                                    ex.methods[METHOD].prior, expanded_q, pool)
                for ex, expanded_q, pool in needs_selection
            ]
            selection_raw = generator.generate_batch(
                selection_prompts, args.max_new_tokens_answer, desc=f"oracle-select-hop{hop_j}",
            )
            for (ex, expanded_q, pool), raw in zip(needs_selection, selection_raw):
                choice = parse_json_choice(raw)
                if choice is None:
                    choice = 0
                choice = max(0, min(len(pool) - 1, choice))
                chosen_para = pool[choice]
                gold_pools[ex.eid] = [p for p in gold_pools[ex.eid] if p is not chosen_para]
                selected.append((ex, expanded_q, chosen_para))

        # Fallback: top-1 cosine retrieval (gt: malformed gold record; bart: pool exhausted).
        fallback_hits: list[tuple[Any, str, dict[str, Any]]] = []
        for ex, expanded_q in needs_fallback:
            cands = embed_retrieval(expanded_q, ex.paragraphs, 1, args.cos_model)
            if cands:
                fallback_hits.append((ex, expanded_q, cands[0][0]))

        answer_meta: list[tuple[Any, str, dict[str, Any], str]] = (
            [(ex, q, p, "gold_direct") for ex, q, p in direct_hits]
            + [(ex, q, p, "gold_selected") for ex, q, p in selected]
            + [(ex, q, p, "fallback_top1") for ex, q, p in fallback_hits]
        )
        n_gold_hops += len(direct_hits) + len(selected)
        n_fallback_hops += len(fallback_hits)

        answer_prompts = [
            prompt_short_answer_with_context(ex.q_main, ex.methods[METHOD].hop_steps,
                                              ex.methods[METHOD].prior, expanded_q, chosen_para)
            for ex, expanded_q, chosen_para, _source in answer_meta
        ]

        print(f"[hop {hop_j}] {len(answer_meta)} examples "
              f"({len(direct_hits)} gold_direct, {len(selected)} gold_selected, "
              f"{len(fallback_hits)} fallback)", flush=True)

        answer_raw = generator.generate_batch(
            answer_prompts, args.max_new_tokens_answer, desc=f"oracle-answer-hop{hop_j}",
        ) if answer_prompts else []

        for (ex, expanded_q, chosen_para, source), raw in zip(answer_meta, answer_raw):
            ms = ex.methods[METHOD]
            sub_ans = normalize_short_answer(raw)
            ev_text = (chosen_para.get("paragraph_text") or "").strip()
            ms.prior.append(sub_ans)
            ms.hop_steps.append((expanded_q, ev_text))
            ex.hop_results[hop_j - 1][METHOD] = {
                "expanded_subq": expanded_q,
                "evidence_source": source,
                "chosen_para_id": int(chosen_para.get("idx", -1)),
                "sub_answer": sub_ans,
            }

    # Final answer, same prompt/system-prompt as production wavefront's final reader.
    final_tasks = [(ex, ex.methods[METHOD]) for ex in examples if ex.methods[METHOD].hop_steps]
    final_prompts = [
        build_final_reader_cot_prompt(ex.q_main, ms.hop_steps, ms.prior) for ex, ms in final_tasks
    ]
    final_raw = generator.generate_chat_batch(
        FINAL_READER_SYSTEM_PROMPT, final_prompts, args.max_new_tokens_final, desc="oracle-final-reader",
    )

    out_path = args.out or (HERE / "results" / f"oracle_reader_{args.dataset}_{args.split}_{args.decompose_mode}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_scored = n_em = 0
    n_f1 = 0.0
    with out_path.open("w", encoding="utf-8") as f:
        for (ex, ms), raw in zip(final_tasks, final_raw):
            fallback = ms.prior[-1] if ms.prior else ""
            final_answer = parse_final_answer_from_cot(raw, fallback=fallback)
            em, f1 = judge_answer_official(final_answer, ex.row)
            n_scored += 1
            n_em += int(em or 0)
            n_f1 += float(f1 or 0.0)
            f.write(json.dumps({
                "id": ex.eid, "K": ex.K, "question": ex.q_main,
                "final_answer": final_answer, "gold_answer": ex.row.get("answer"),
                "em": em, "f1": f1,
                "evidence_sources": [ex.hop_results[j].get(METHOD, {}).get("evidence_source") for j in range(ex.K)],
                "prior": ms.prior,
            }, ensure_ascii=False) + "\n")

    total_hops = n_gold_hops + n_fallback_hops
    gold_cov = n_gold_hops / total_hops if total_hops else 0.0
    print(f"\nOracle-evidence reader done: {n_scored} examples -> {out_path}")
    print(f"Gold coverage: {n_gold_hops}/{total_hops} hops ({gold_cov:.4f}) used real gold evidence "
          f"({n_fallback_hops} hops fell back to top-1 cosine)")
    if n_scored:
        print(f"EM={n_em / n_scored:.4f} F1={n_f1 / n_scored:.4f}")


if __name__ == "__main__":
    main()
