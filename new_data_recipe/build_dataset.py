#!/usr/bin/env python3
"""
Step 10 (see update_doc/0907/0907update.md section 9): end-to-end driver, chunked +
incremental + resumable. Assembles recipe 1/2/3/4 traces for MuSiQue train cases, generates
final answers, extracts raw hidden states, and writes the three output files:
  traces_v2/{dataset}/{split}.jsonl
  pairs_v2/{dataset}/{split}.jsonl
  hidden_states_v2/{dataset}/{split}/{trace_id}.npz

Cases are processed in chunks of --chunk-size (default 200). Each chunk's traces/pairs are
APPENDED to the output jsonl files as soon as that chunk finishes (not buffered for the whole
run) -- so a crash mid-run only loses at most one chunk's worth of GPU work, and stats.json is
rewritten (cumulative) after every chunk.

Resumable: on startup, any case_id that already has a recipe-2 ("2_real_baseline") trace in the
existing traces_v2.jsonl (same --out-dir) is treated as already done and skipped entirely --
recipe 2 is unconditionally generated for every case and only ever appears in the output file
once its whole chunk has finished, so its presence is a reliable per-case completion marker.
Re-running the exact same command against the same --out-dir after a crash (or to extend an
earlier smaller run to more cases) picks up where it left off instead of redoing finished cases.

Usage (small smoke test first):
  CUDA_VISIBLE_DEVICES=0 python build_dataset.py --limit-per-k 5 --out-dir output_small_test
Usage (full run, resumable):
  CUDA_VISIBLE_DEVICES=1 python build_dataset.py --limit-per-k 0 --out-dir output_full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACES_DIR = HERE.parent / "traces"
RETRIEVAL_DIR = HERE.parent / "retrieval"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TRACES_DIR))
sys.path.insert(0, str(RETRIEVAL_DIR))

from dataset_loaders import (  # noqa: E402
    get_decompose_path, get_decompose_enhance_path, load_decompose_bundle, load_raw_index,
)
from trace_evidence import gold_evidence_musique, gold_hop_answers_musique, get_ranker  # noqa: E402
from run_retrieval_exp import select_example_ids  # noqa: E402
from run_retrieval_exp_wavefront import (  # noqa: E402
    BatchVllmGenerator, load_gate_model, warmup_gate_model_memory,
)
from recipe2_core import add_hidden_states  # noqa: E402
from recipe2_batch import (  # noqa: E402
    run_recipe2_batch, add_final_answers_batch, build_recipe1_gold_trace, should_skip_recipe1,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="musique")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit-per-k", type=int, default=5)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=HERE / "output_small_test")
    ap.add_argument("--chunk-size", type=int, default=200,
                     help="cases per chunk -- traces/pairs are appended to disk after each "
                          "chunk finishes, bounding how much work a crash can lose")
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    return ap.parse_args()


def load_done_case_ids(traces_path: Path) -> set[str]:
    """A case counts as done iff its recipe-2 trace was already written to traces_path. Recipe 2
    is generated unconditionally for every case and traces_path is only ever appended to once a
    whole chunk has finished start-to-finish (see main()), so this is a reliable per-case
    completion marker even if the process was killed mid-chunk last time."""
    done: set[str] = set()
    if not traces_path.is_file():
        return done
    with traces_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("recipe") == "2_real_baseline":
                done.add(r["case_id"])
    return done


def load_stats(stats_path: Path) -> dict:
    if stats_path.is_file():
        try:
            return json.loads(stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"n_cases": 0, "n_traces": 0, "n_pairs": 0, "recipe_counts": {}, "wall_clock_sec": 0.0}


def strip_case(c):
    return {k: c[k] for k in ("case_id", "question", "sub_qs", "gold_idxs", "gold_texts", "paragraphs")}


def process_chunk(
    chunk_cases: list[dict],
    hop_answers_by_id: dict[str, list[str]],
    *,
    generator, ranker, args,
) -> list[dict]:
    """Same recipe 1/2/3/4 + cross-recipe dedup logic as before, scoped to one chunk of cases.
    Dedup signatures are always (case_id, committed_idx tuple) -- since a case never spans two
    chunks, running this per-chunk is exactly equivalent to running it over the whole dataset at
    once."""
    # --- recipe 2: real baseline trace for every case in this chunk ---
    r2_traces = run_recipe2_batch(
        [strip_case(c) for c in chunk_cases], generator=generator, ranker=ranker,
        cos_model=args.cos_model, retrieve_k=args.retrieve_k, max_new_tokens=args.max_new_tokens,
        is_natural_seed=True,
    )
    all_traces = list(r2_traces)

    # --- recipe 1: gold backfill, only where recipe2 wasn't fully correct ---
    for c, r2 in zip(chunk_cases, r2_traces):
        if not should_skip_recipe1(r2):
            all_traces.append(build_recipe1_gold_trace(c, hop_answers_by_id[c["case_id"]]))

    # --- recipe 3: single forced-wrong hop, rest REAL/natural continuation -- one variant per
    # hop position, for EVERY case regardless of whether recipe2 itself succeeded.
    r3_cases = []
    r3_forced_hops = {}
    for c in chunk_cases:
        K = len(c["sub_qs"])
        for h in range(1, K + 1):
            idx = len(r3_cases)
            r3_cases.append(strip_case(c))
            r3_forced_hops[idx] = h
    if r3_cases:
        r3_traces = run_recipe2_batch(
            r3_cases, generator=generator, ranker=ranker, cos_model=args.cos_model,
            retrieve_k=args.retrieve_k, max_new_tokens=args.max_new_tokens,
            forced_hops=r3_forced_hops, trace_id_suffix="recipe3_forced", recipe_name="3_forced_wrong",
        )
        for idx, t in enumerate(r3_traces):
            old_trace_id = t["trace_id"]
            new_trace_id = f"{old_trace_id}_h{r3_forced_hops[idx]}"
            t["trace_id"] = new_trace_id
            for pair in t["pairs"]:
                pair["trace_id"] = new_trace_id
                pair["pair_id"] = pair["pair_id"].replace(old_trace_id, new_trace_id, 1)
            for hop in t["hops"]:
                if "pair_id" in hop:
                    hop["pair_id"] = hop["pair_id"].replace(old_trace_id, new_trace_id, 1)
        all_traces.extend(r3_traces)

    # --- recipe 4: clean-prefix sweep, a=1..K-1, for EVERY case ---
    r4_cases = []
    r4_clean_upto = {}
    r4_gold_answers = {}
    r4_a = {}
    for c in chunk_cases:
        K = len(c["sub_qs"])
        for a in range(1, K):
            idx = len(r4_cases)
            r4_cases.append(strip_case(c))
            r4_clean_upto[idx] = a
            r4_gold_answers[idx] = hop_answers_by_id[c["case_id"]]
            r4_a[idx] = a
    if r4_cases:
        r4_traces = run_recipe2_batch(
            r4_cases, generator=generator, ranker=ranker, cos_model=args.cos_model,
            retrieve_k=args.retrieve_k, max_new_tokens=args.max_new_tokens,
            clean_prefix_upto=r4_clean_upto, clean_prefix_gold_hop_answers=r4_gold_answers,
            trace_id_suffix="recipe4_cleanprefix", recipe_name="4_clean_prefix",
        )
        for idx, t in enumerate(r4_traces):
            old_trace_id = t["trace_id"]
            new_trace_id = f"{old_trace_id}_a{r4_a[idx]}"
            t["trace_id"] = new_trace_id
            for pair in t["pairs"]:
                pair["trace_id"] = new_trace_id
                pair["pair_id"] = pair["pair_id"].replace(old_trace_id, new_trace_id, 1)
            for hop in t["hops"]:
                if "pair_id" in hop:
                    hop["pair_id"] = hop["pair_id"].replace(old_trace_id, new_trace_id, 1)
        all_traces.extend(r4_traces)

    # --- dedup within this chunk (never needs to see other chunks: signatures are per-case) ---
    seen_sig = {}
    deduped = []
    for t in all_traces:
        sig = (t["case_id"], tuple(h["committed_idx"] for h in t["hops"]))
        if sig in seen_sig:
            continue
        seen_sig[sig] = t["trace_id"]
        deduped.append(t)
    return deduped


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()

    decompose_path = get_decompose_path(args.dataset, args.split, mode="gt")
    enhance_path = get_decompose_enhance_path(args.dataset, args.split) if args.dataset == "musique" else None
    decomp, raw_id_map = load_decompose_bundle(
        decompose_path, enhance_path=enhance_path, include_enhance=True,
    )
    raw_index = load_raw_index(args.dataset, args.split)

    ids = sorted(decomp.keys())
    ids = select_example_ids(ids, decomp, limit=0, limit_per_k=args.limit_per_k, seed=args.sample_seed)

    traces_path = args.out_dir / "traces_v2" / args.dataset / f"{args.split}.jsonl"
    pairs_path = args.out_dir / "pairs_v2" / args.dataset / f"{args.split}.jsonl"
    hs_dir = args.out_dir / "hidden_states_v2" / args.dataset / args.split
    stats_path = args.out_dir / "stats.json"
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)

    done_case_ids = load_done_case_ids(traces_path)
    if done_case_ids:
        print(f"resume: {len(done_case_ids)} cases already done in {traces_path}, skipping them")

    cases = []
    hop_answers_by_id = {}
    record_by_case_id = {}
    for did in ids:
        if did in done_case_ids:
            continue
        raw_id = raw_id_map.get(did, did)
        record = raw_index.get(raw_id)
        if record is None:
            continue
        sub_qs = decomp[did]
        gold_texts, gold_idxs = gold_evidence_musique(record)
        hop_answers = gold_hop_answers_musique(record)
        k = min(len(sub_qs), len(gold_idxs), len(hop_answers))
        if k < 1:
            continue
        cases.append({
            "case_id": did, "question": record.get("question", ""), "sub_qs": sub_qs[:k],
            "gold_idxs": gold_idxs[:k], "gold_texts": gold_texts[:k],
            "paragraphs": record.get("paragraphs") or [],
        })
        hop_answers_by_id[did] = hop_answers[:k]
        record_by_case_id[did] = record
    print(f"{len(cases)} cases remaining to process (of {len(ids)} selected, "
          f"limit_per_k={args.limit_per_k})")

    if not cases:
        print("nothing to do -- all selected cases already done.")
        return

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
    ranker = get_ranker(args.cos_model)

    stats = load_stats(stats_path)
    recipe_counts = Counter(stats.get("recipe_counts", {}))
    n_cases_total = stats.get("n_cases", 0)
    n_traces_total = stats.get("n_traces", 0)
    n_pairs_total = stats.get("n_pairs", 0)
    prior_wall_clock_sec = stats.get("wall_clock_sec", 0.0)

    n_chunks = (len(cases) + args.chunk_size - 1) // args.chunk_size
    for ci in range(n_chunks):
        chunk_cases = cases[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        print(f"\n=== chunk {ci + 1}/{n_chunks}: {len(chunk_cases)} cases "
              f"({chunk_cases[0]['case_id']} .. {chunk_cases[-1]['case_id']}) ===")

        chunk_traces = process_chunk(chunk_cases, hop_answers_by_id, generator=gen, ranker=ranker, args=args)

        gold_records = [record_by_case_id[t["case_id"]] for t in chunk_traces]
        add_final_answers_batch(chunk_traces, gold_records, gen)

        for t in chunk_traces:
            add_hidden_states(
                t, gate_model=gate_model, gate_tokenizer=gate_tokenizer, layer=args.layer,
                npz_path=hs_dir / f"{t['trace_id']}.npz",
            )

        # Build both files' text fully in memory first, then do ONE write()+fsync() per file --
        # narrows the "killed mid-chunk" vulnerable window from ~one syscall per trace/pair line
        # down to two syscalls total. Pairs are flushed before traces, since traces_v2.jsonl
        # (specifically its recipe-2 lines) is what load_done_case_ids() uses to decide a case is
        # done -- so a crash between the two writes can at worst cause one chunk's pairs to be
        # duplicated on the next run's redo of that chunk, never a case marked done while missing
        # its pairs.
        trace_lines = []
        pair_lines = []
        chunk_n_pairs = 0
        for t in chunk_traces:
            pairs = t.pop("pairs", [])
            recipe_counts[t["recipe"]] += 1
            trace_lines.append(json.dumps(t, ensure_ascii=False))
            for p in pairs:
                pair_lines.append(json.dumps(p, ensure_ascii=False))
                chunk_n_pairs += 1

        if pair_lines:
            with pairs_path.open("a", encoding="utf-8") as pf:
                pf.write("\n".join(pair_lines) + "\n")
                pf.flush()
                os.fsync(pf.fileno())
        if trace_lines:
            with traces_path.open("a", encoding="utf-8") as tf:
                tf.write("\n".join(trace_lines) + "\n")
                tf.flush()
                os.fsync(tf.fileno())

        n_cases_total += len(chunk_cases)
        n_traces_total += len(chunk_traces)
        n_pairs_total += chunk_n_pairs
        stats = {
            "n_cases": n_cases_total, "n_traces": n_traces_total, "n_pairs": n_pairs_total,
            "recipe_counts": dict(recipe_counts),
            "wall_clock_sec": round(prior_wall_clock_sec + (time.perf_counter() - t0), 1),
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"  chunk done: +{len(chunk_traces)} traces, +{chunk_n_pairs} pairs "
              f"(cumulative: {n_cases_total} cases, {n_traces_total} traces, {n_pairs_total} pairs)")

    print(f"\nWrote traces -> {traces_path}")
    print(f"Wrote pairs -> {pairs_path}")
    print(f"Wrote hidden states -> {hs_dir}")
    print(f"stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
