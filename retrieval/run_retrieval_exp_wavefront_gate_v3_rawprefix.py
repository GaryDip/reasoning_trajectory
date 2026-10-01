#!/usr/bin/env python3
"""
Gate v3 wavefront, with a SEPARATE raw (unresolved) accumulated prefix used only for gate
scoring (see conversation / 0727 update doc section 5).

Confirmed by inspecting the actual training data: hidden_states/pilot_multilayer (gate
v2/v3's training source, extracted from traces/merged/musique/*.jsonl's reasoning_trace
field) NEVER resolves a sub-question's "[Answer N]" back-reference -- about 49% of hops
contain the literal, unresolved placeholder in gate v2/v3's own training text. But
`run_retrieval_exp_wavefront_gate_v3.py` DOES resolve it at inference (via
expand_hop_template + the per-hop generated short answer) before building both the
retrieval query AND the text gate v3 scores -- a real train/inference mismatch.

An earlier attempt (`run_retrieval_exp_wavefront_gate_v3_decoupled.py`, since replaced by
this file) fixed this by using the raw sub-question EVERYWHERE, including for the retrieval
query itself, and fully decoupling retrieval from generation. That regressed hard: full
musique dev recall@1 -5.93pt, chain -9.80pt (see 0727 update doc 5.4) -- per-hop breakdown
showed hop 1 (which never has a placeholder to begin with) identical between the two
versions down to the decimal, while every later hop got substantially worse, confirming the
damage was entirely on the retrieval side: BGE is a general-purpose embedder with no
exposure to this pipeline, so an unresolved "[Answer N]" in the query genuinely degrades
candidate-pool quality for bridge-style hops.

This script fixes the mismatch WITHOUT touching retrieval, by keeping two separate
accumulated step-lists per beam path instead of one:

  - `hop_steps`  (expanded sub-questions) -- used for retrieval-query resolution, the
    per-hop short-answer prompt, and the final reader -- IDENTICAL to what
    run_retrieval_exp_wavefront_gate_v3.py already does, unchanged.
  - `gate_hop_steps` (raw, never-expanded sub-questions) -- used ONLY to build the
    accumulated prefix gate v3 scores against, so the text gate v3 sees at inference (from
    hop 1 through the current hop) matches its training distribution exactly, not just at
    the current hop but across the whole accumulated prefix.

Retrieval, the per-hop short-answer generation, and the final reader are otherwise
IDENTICAL to run_retrieval_exp_wavefront_gate_v3.py -- this is a much narrower change than
the decoupled attempt: only the text fed to gate v3 for scoring changes, nothing about how
candidates are retrieved or how answers are generated.

Also applies BGE's own documented query-side instruction prefix ("Represent this sentence
for searching relevant passages: ") before calling retrieval/run_retrieval_exp.py's shared
embed_retrieval() -- confirmed by reading that function that it encodes the query as-is,
with no instruction prefix, so every method reusing it (baseline, gate v2, gate v3, this
script's own earlier version) has been retrieving without it all along. Only the query text
passed INTO embed_retrieval changes (via --query-instruction, pass '' to reproduce the old
unprefixed behavior) -- the shared function itself is untouched.

Needs (train first if missing): gate/gate_v3/artifacts_pooled_v3/{j0..j3}.joblib

Reuses (imports, does not copy) retrieval/run_retrieval_exp.py,
retrieval/run_retrieval_exp_wavefront.py, and this directory's own
run_retrieval_exp_wavefront_gate_v3.py (load_gate_v3_artifacts/score_gate_v3) -- none of
those files are modified.

Usage:
  python run_retrieval_exp_wavefront_gate_v3_rawprefix.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RETRIEVAL_DIR = HERE
PROJECT_ROOT = HERE.parent
GATE_V3_ARTIFACTS_DIR = PROJECT_ROOT / "gate" / "gate_v3" / "artifacts_pooled_v3"
TRACES_DIR = PROJECT_ROOT / "traces"
sys.path.insert(0, str(TRACES_DIR))
sys.path.insert(0, str(RETRIEVAL_DIR))

from run_retrieval_exp import (  # noqa: E402
    HOTPOT_DEV_FILE,
    K_BUCKETS,
    MUSIQUE_DIR,
    TWOWIKI_DEV_FILE,
    AnswerAccum,
    MetricAccum,
    _get_pipeline_helpers,
    build_trace_prefix,
    colbert_retrieval,
    embed_retrieval,
    embed_retrieval_against_corpus,
    find_gold_rank,
    find_gold_rank_set,
    load_dataset_records,
    load_decompose_index,
    normalize_short_answer,
    prepare_sub_questions,
    resolve_decompose_file,
    resolve_output_dir,
    sanitize_run_tag,
    select_example_ids,
)
from build_global_corpus import load_global_corpus  # noqa: E402
from run_retrieval_exp_wavefront import (  # noqa: E402
    FINAL_READER_SYSTEM_PROMPT,
    BatchVllmGenerator,
    batch_last_hidden,
    build_final_reader_cot_prompt,
    build_final_reader_cot_prompt_comparison_hint,
    build_final_reader_cot_prompt_comparison_positive,
    load_gate_model,
    parse_final_answer_from_cot,
    prompt_short_answer_with_context,
    prompt_short_answer_with_context_type_match,
)

FINAL_READER_PROMPT_BUILDERS = {
    "default": build_final_reader_cot_prompt,
    "comparison_hint": build_final_reader_cot_prompt_comparison_hint,
    "comparison_positive": build_final_reader_cot_prompt_comparison_positive,
}

SHORT_ANSWER_PROMPT_BUILDERS = {
    "default": prompt_short_answer_with_context,
    "type_match": prompt_short_answer_with_context_type_match,
}
from run_retrieval_exp_wavefront_gate_v3 import load_gate_v3_artifacts, score_gate_v3  # noqa: E402
from trace_evidence import gold_evidence_musique  # noqa: E402
from trace_format import escape_double_quotes  # noqa: E402


def minmax_normalize_candidates(candidates: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
    """
    Rescale a single hop's candidate scores to [0, 1] within that pool, in place of using
    the raw score directly in `rank_key = emb_score - lambda_gate * gate_score`.

    Needed for --retriever colbert: ColBERT's MaxSim score is a SUM over query tokens, so
    longer queries mechanically get larger raw scores than shorter ones -- unlike cosine
    similarity (bounded, roughly comparable in scale to gate_score's [0,1] probability),
    ColBERT's raw score is neither bounded nor comparable across different queries. Without
    this, `lambda_gate * gate_score` (at most lambda_gate, a value like 0.5) is negligible
    next to a raw ColBERT score in the tens, and the gate signal has essentially no effect
    on the rerank. Min-max normalizing within each hop's own candidate pool (not globally)
    puts the retrieval signal back on the same [0,1] scale as gate_score, so lambda_gate is
    meaningful again -- though the optimal lambda_gate for this combination still likely
    differs from the value tuned for raw cosine scores and needs its own sweep.

    Only applied on the colbert path; the cosine path's raw scores are left untouched (BGE's
    lambda_gate=0.50 was tuned against them directly).
    """
    if not candidates:
        return candidates
    scores = [s for _, s in candidates]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [(p, 0.5) for p, _ in candidates]
    return [(p, (s - lo) / (hi - lo)) for p, s in candidates]


def warmup_gate_model_memory(gate_model, gate_tokenizer, batch_size: int, warmup_tokens: int = 4096) -> None:
    """
    PyTorch's caching allocator only cudaMalloc's what a forward pass actually needs, and
    the text length fed to the gate model grows across hops (build_trace_prefix accumulates
    the whole raw prefix, not just the current hop) -- so peak GPU memory for this model is
    only reached late in a run, at which point a concurrent process on a shared card may
    have already taken the headroom this run would need. Running one oversized dummy forward
    pass up front (well above any realistic per-hop text length) forces PyTorch to grab that
    memory while it's available; the caching allocator holds onto it for the rest of the
    process's life (as long as torch.cuda.empty_cache() is never called), so later, shorter
    or longer-but-still-under-this-cap batches reuse it instead of issuing a fresh cudaMalloc
    that could fail mid-run.

    4096 is sized off measurements across all three datasets this script runs on, not just
    musique: musique's own cumulative K=4 trace text (traces/merged/musique/dev.jsonl) tops
    out at 1554 tokens (p999 1189) -- but 2wiki and hotpot's individual candidate paragraphs
    alone can reach 1797 / 1980 tokens (p999 870 / 678 per paragraph), so a K=4 cumulative
    prefix built from several such paragraphs can plausibly approach ~3800-4000 tokens even
    without hitting the extreme per-paragraph max at every hop. An earlier version of this
    warm-up used 2048, sized off musique alone -- too small once 2wiki/hotpot's longer
    per-paragraph lengths are accounted for.
    """
    import torch

    dev = next(gate_model.parameters()).device
    dummy_text = "warmup " * warmup_tokens
    inputs = gate_tokenizer(
        [dummy_text] * batch_size, return_tensors="pt", truncation=True,
        max_length=warmup_tokens, padding=True,
    ).to(dev)
    with torch.no_grad():
        gate_model(**inputs, output_hidden_states=True)
    print(f"Gate model warm-up done: batch={batch_size} tokens<={warmup_tokens} on {dev}", flush=True)


@dataclass
class BeamPath:
    hop_steps: list[tuple[str, str]] = field(default_factory=list)       # (expanded_q, evidence) -- reader-facing
    gate_hop_steps: list[tuple[str, str]] = field(default_factory=list)  # (raw_sub_q, evidence) -- gate-scoring only
    prior: list[str] = field(default_factory=list)
    para_ids: list[int] = field(default_factory=list)
    rank_key: float = 0.0
    gate_v3_score: float = 0.0


@dataclass
class ExampleState:
    eid: str
    row: dict[str, Any]
    q_main: str
    paragraphs: list[dict[str, Any]]
    sub_questions: list[str]
    gold_idxs: list[int]
    K: int
    beams: list[BeamPath]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique", "2wiki", "hotpot"], default="musique")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--twowiki-file", type=Path, default=TWOWIKI_DEV_FILE)
    ap.add_argument("--hotpot-file", type=Path, default=HOTPOT_DEV_FILE)
    ap.add_argument("--decompose-mode", choices=["gt", "bart_decompose"], default="bart_decompose")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--artifacts-dir", type=Path, default=GATE_V3_ARTIFACTS_DIR)
    ap.add_argument("--lambda-gate", type=float, default=0.60,
                     help="Updated 0828 from 0.50 -> 0.60: a 3000-example BART-decompose train "
                          "subset sweep (0.10-0.80) found EM/F1 peaking at 0.60-0.65, and a "
                          "direct 3-dataset dev re-check at 0.60 (current rawprefix+"
                          "comparison_hint config) confirmed it -- EM/F1 improve on 2wiki/hotpot "
                          "(+0.4-0.7pp) with only a small musique regression (-0.4pp), net a "
                          "more broadly-robust choice than 0.50 (which was originally tuned "
                          "mostly against musique, see 0727 update doc section 6).")
    ap.add_argument("--beam-width", type=int, default=1)
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--retriever", choices=("cosine", "colbert"), default="cosine",
                     help="'colbert' uses run_retrieval_exp.py's colbert_retrieval() -- "
                          "in-memory MaxSim over this example's own candidate paragraphs, "
                          "no offline index needed (candidate pools here are ~10-20 "
                          "paragraphs, not a shared corpus). Never actually run before; "
                          "needs `pip install colbert-ai`.")
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--colbert-model", default="colbert-ir/colbertv2.0")
    ap.add_argument("--retrieval-scope", choices=("distractor", "global"), default="distractor",
                     help="'distractor' (default, unchanged behavior): retrieve only within this "
                          "example's own gold+distractor paragraphs. 'global': retrieve against a "
                          "shared corpus pooled across the whole dataset split (see "
                          "build_global_corpus.py) -- a much harder, more realistic setting since "
                          "the gold passage is no longer guaranteed to be in a small pre-filtered "
                          "pool. Requires --corpus-dir. --retriever must be 'cosine' (colbert's "
                          "in-memory MaxSim was never built for a corpus this size).")
    ap.add_argument("--corpus-dir", type=Path, default=None,
                     help="directory from build_global_corpus.py (corpus_meta.jsonl + "
                          "corpus_emb.npy) -- required when --retrieval-scope global, ignored "
                          "otherwise. e.g. global_corpus/musique_dev/")
    ap.add_argument(
        "--query-instruction",
        default="Represent this sentence for searching relevant passages: ",
        help="BGE's own documented instruction prefix for the QUERY side of asymmetric "
             "retrieval (short query -> long passage) -- retrieval/run_retrieval_exp.py's "
             "shared embed_retrieval() never applies this (confirmed by reading it: query "
             "text is encoded as-is), so every method reusing it unprefixed has been doing "
             "so all along. Applied here only to the query text passed into embed_retrieval, "
             "not to embed_retrieval() itself -- that shared function is untouched. Pass "
             "'' to reproduce the old unprefixed behavior for comparison.",
    )
    ap.add_argument("--answerable-only", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--limit-per-k", type=int, default=0,
                     help="Stratified sample: up to N examples per gold hop count K in "
                          "{1,2,3,4} (see run_retrieval_exp.py::select_example_ids). Takes "
                          "priority over --limit when > 0. Intended for sweeping "
                          "--lambda-gate on a balanced-by-K subset (e.g. of --split train) "
                          "without re-using the same data the final report evaluates on.")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--results-root", type=Path, default=HERE / "results")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--gate-device", default="cuda:0")
    ap.add_argument("--hidden-batch-size", type=int, default=8)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens-answer", type=int, default=64)
    ap.add_argument("--max-new-tokens-final", type=int, default=128)
    ap.add_argument("--final-reader-prompt", choices=list(FINAL_READER_PROMPT_BUILDERS), default="comparison_hint",
                     help="Default 'comparison_hint' (the D-group setting since 0831; 'default' is the "
                          "original prompt without it). 'comparison_hint' adds one instruction telling the reader that "
                          "for comparison-style questions, the per-hop answers are inputs "
                          "to a comparison it still has to perform, not the final answer "
                          "verbatim -- see build_final_reader_cot_prompt_comparison_hint's "
                          "docstring for the case-study finding that motivated this.")
    ap.add_argument("--short-answer-prompt", choices=list(SHORT_ANSWER_PROMPT_BUILDERS), default="default",
                     help="'type_match' adds one instruction requiring the per-hop short "
                          "answer's type to match the subquestion's interrogative word "
                          "(where/when/who/how many) instead of defaulting to the nearest "
                          "number in the evidence -- see "
                          "prompt_short_answer_with_context_type_match's docstring for the "
                          "case-study finding that motivated this.")
    return ap.parse_args()


def main() -> None:
    run_t0 = time.perf_counter()
    args = parse_args()
    if args.retrieval_scope == "global":
        if args.corpus_dir is None:
            raise SystemExit("--retrieval-scope global requires --corpus-dir (see build_global_corpus.py)")
        if args.retriever != "cosine":
            raise SystemExit("--retrieval-scope global only supports --retriever cosine "
                              "(colbert's in-memory MaxSim was never built for a corpus this size)")
    args.decompose_file = resolve_decompose_file(
        args.decompose_mode, args.split, args.decompose_file, args.dataset
    )

    ds_args = argparse.Namespace(
        dataset=args.dataset, musique_dir=args.musique_dir, split=args.split,
        twowiki_file=args.twowiki_file, hotpot_file=args.hotpot_file,
    )
    records = load_dataset_records(ds_args)
    decompose_idx = load_decompose_index(args.decompose_file)
    ids = sorted(set(records) & set(decompose_idx))
    if args.answerable_only:
        ids = [eid for eid in ids if records[eid].get("answerable", True)]
    ids = select_example_ids(
        ids, decompose_idx, limit=args.limit, limit_per_k=args.limit_per_k, seed=args.sample_seed
    )

    tag = args.run_tag or f"gate_v3_rawprefix_{args.dataset}_bw{args.beam_width}"
    args.out_dir = resolve_output_dir(
        out_dir=args.out_dir, results_root=args.results_root, run_tag=tag,
        decompose_mode=args.decompose_mode, methods=["gate_v3_rawprefix"],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{sanitize_run_tag(args.run_tag)}" if args.run_tag else ""
    out_cases = args.out_dir / f"retrieval_cases_{args.split}{tag_suffix}.jsonl"
    out_metrics = args.out_dir / f"retrieval_exp_{args.split}{tag_suffix}.json"
    print(f"Output directory: {args.out_dir}")

    (expand_hop_template, *_rest, judge_answer_official) = _get_pipeline_helpers()

    gate_v3_artifacts = load_gate_v3_artifacts(args.artifacts_dir)
    print(f"Gate v3: {args.artifacts_dir} (j -> layer: "
          f"{ {j: art['layer'] for j, art in gate_v3_artifacts.items()} })")

    print(f"Examples: {len(ids)}  beam_width={args.beam_width}  decompose={args.decompose_file}")
    examples: list[ExampleState] = []
    for eid in ids:
        row = records[eid]
        sub_qs = prepare_sub_questions(decompose_idx.get(eid) or [], decompose_mode=args.decompose_mode)
        if not sub_qs:
            continue
        _gold_texts, gold_idxs = gold_evidence_musique(row)
        examples.append(ExampleState(
            eid=eid, row=row, q_main=(row.get("question") or "").strip(),
            paragraphs=row.get("paragraphs") or [], sub_questions=sub_qs,
            gold_idxs=gold_idxs, K=len(sub_qs), beams=[BeamPath()],
        ))

    # --- global retrieval scope: swap each example's own paragraph pool for one shared,
    # precomputed corpus (see build_global_corpus.py), and remap gold_idxs (currently indices
    # local to that example's own `paragraphs`) into the corpus's global id space via
    # (title, paragraph_text) lookup -- everything downstream (gate scoring, find_gold_rank,
    # MetricAccum/ChainAccum) only ever compares `para["idx"]` ints, so once ex.paragraphs and
    # ex.gold_idxs both speak the corpus's global ids, no other code needs to change.
    corpus_paragraphs = corpus_doc_emb = None
    if args.retrieval_scope == "global":
        print(f"Loading global corpus from {args.corpus_dir} ...", flush=True)
        corpus_paragraphs, corpus_doc_emb, title_text_to_idx = load_global_corpus(args.corpus_dir)
        print(f"  {len(corpus_paragraphs)} unique paragraphs, embedding {corpus_doc_emb.shape}")
        n_remapped = 0
        n_missing = 0
        for ex in examples:
            local_by_idx = {int(p.get("idx", -1)): p for p in ex.paragraphs}
            new_gold_idxs: list[int | None] = []
            for gi in ex.gold_idxs:
                local_para = local_by_idx.get(gi)
                key = (
                    (local_para.get("title") or "").strip(),
                    (local_para.get("paragraph_text") or "").strip(),
                ) if local_para is not None else None
                global_idx = title_text_to_idx.get(key) if key is not None else None
                if global_idx is None:
                    n_missing += 1
                    new_gold_idxs.append(None)
                else:
                    n_remapped += 1
                    new_gold_idxs.append(global_idx)
            ex.gold_idxs = new_gold_idxs
            # ex.paragraphs is deliberately left as this example's own local pool -- it's
            # unused in global scope (the retrieval branch below reads corpus_paragraphs/
            # corpus_doc_emb directly instead), and overwriting it with the whole corpus would
            # make embed_retrieval() re-encode 21k+ docs on every single hop call (the exact
            # per-call-re-encode cost this whole global-corpus path exists to avoid).
        print(f"  gold remap: {n_remapped} hops mapped to a global idx, {n_missing} not found "
              f"in the corpus (should be ~0 -- corpus was built from this same split's own data)")

    print("Loading Llama model for hidden-state scoring ...", flush=True)
    gate_model, gate_tokenizer = load_gate_model(args)
    warmup_gate_model_memory(gate_model, gate_tokenizer, args.hidden_batch_size)
    print("Loading vLLM generator ...", flush=True)
    generator = BatchVllmGenerator(
        args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
    )

    hidden_cache: dict[tuple[int, str], Any] = {}
    max_hops = max((ex.K for ex in examples), default=0)
    oracle_hop_survive: dict[Any, MetricAccum] = {"all": MetricAccum(), **{K: MetricAccum() for K in K_BUCKETS}}
    # Gold's rank in the FULL reranked candidate pool (up to --retrieve-k), computed BEFORE
    # the beam_width truncation below -- with --beam-width 1, oracle_hop_survive's
    # recall@1/@3 degenerate to the same number (only 1 survivor to check), so this is the
    # only place recall@3 means anything: is gold in the top-3 by rank_key, not just
    # whether the single kept candidate happens to be gold.
    full_pool_rank: dict[Any, MetricAccum] = {"all": MetricAccum(), **{K: MetricAccum() for K in K_BUCKETS}}
    # Order-invariant twin of full_pool_rank: same full (pre-beam-width) candidate pool, same
    # rank_key ranking, but the "hit" test is "is ANY of this example's gold passages in the
    # pool" instead of "is THIS hop's specific positionally-assigned gold passage in the pool".
    # Added so recall@1/@3 can be compared against methods whose own decomposition doesn't
    # guarantee a stable hop-position <-> gold-position correspondence (ChainRAG, GRITHopper --
    # neither has anything like MuSiQue's own annotated hop order to align against), using the
    # exact same definition applied to those methods rather than this project's own
    # position-strict convention, which only made sense when comparing against itself.
    full_pool_rank_orderinvariant: dict[Any, MetricAccum] = {"all": MetricAccum(), **{K: MetricAccum() for K in K_BUCKETS}}
    short_answer_prompt_fn = SHORT_ANSWER_PROMPT_BUILDERS[args.short_answer_prompt]
    # Per-hop full reranked pool (paragraph idx, rank_key order, before beam truncation), one
    # list per hop actually run -- written to the case log so ranking-depth metrics such as
    # precision@3 can be computed offline, same as GRITHopper/ChainRAG's logged ranked_titles.
    hop_ranked_by_ex: dict[str, list[list[int]]] = defaultdict(list)

    for hop_j in range(1, max_hops + 1):
        t0 = time.perf_counter()
        print(f"\n=== hop {hop_j}/{max_hops} ===", flush=True)

        art = gate_v3_artifacts.get(hop_j - 1)
        layer = int(art["layer"]) if art is not None else 31

        # Retrieval: UNCHANGED from run_retrieval_exp_wavefront_gate_v3.py -- expanded_q
        # resolves "[Answer N]" via the per-hop short answers generated so far.
        expand_retrieve: list[tuple[ExampleState, int, str, str, list[tuple[dict, float]]]] = []
        for ex in examples:
            if hop_j > ex.K:
                continue
            raw_sq = ex.sub_questions[hop_j - 1]
            for path_idx, path in enumerate(ex.beams):
                expanded_q = expand_hop_template(raw_sq, path.prior)
                if args.retriever == "colbert":
                    # BGE's query-instruction prefix is that model's own asymmetric-retrieval
                    # convention -- ColBERT was never trained with it, so it's only applied
                    # on the cosine path, not here.
                    candidates_all = colbert_retrieval(expanded_q, ex.paragraphs, args.retrieve_k, args.colbert_model)
                    # ColBERT's raw MaxSim score isn't on a [0,1] scale comparable to
                    # gate_score (see minmax_normalize_candidates docstring) -- rescale
                    # within this hop's own candidate pool before it reaches rank_key.
                    candidates_all = minmax_normalize_candidates(candidates_all)
                elif args.retrieval_scope == "global":
                    # scores against the precomputed corpus_doc_emb matrix -- only the query
                    # gets encoded here, not the whole corpus (see embed_retrieval_against_corpus).
                    query_text = f"{args.query_instruction}{expanded_q}"
                    candidates_all = embed_retrieval_against_corpus(
                        query_text, corpus_doc_emb, corpus_paragraphs, args.retrieve_k, args.cos_model
                    )
                else:
                    query_text = f"{args.query_instruction}{expanded_q}"
                    candidates_all = embed_retrieval(query_text, ex.paragraphs, args.retrieve_k, args.cos_model)
                if not candidates_all:
                    continue
                expand_retrieve.append((ex, path_idx, raw_sq, expanded_q, candidates_all))
        print(f"[hop {hop_j}] {len(expand_retrieve)} (example, path) contexts", flush=True)

        # Gate-scoring text: built from gate_hop_steps (raw sub-questions, all prior hops
        # included) + the CURRENT hop's raw_sq -- matches gate v3's training distribution
        # end to end, not just at this one hop.
        texts: list[str] = []
        owners: list[tuple[int, int]] = []  # (context index, slot); slot=-1 -> prefix_before
        for ctx_idx, (ex, path_idx, raw_sq, expanded_q, candidates_all) in enumerate(expand_retrieve):
            parent = ex.beams[path_idx]
            prefix_before = build_trace_prefix(ex.q_main, parent.gate_hop_steps)
            texts.append(prefix_before)
            owners.append((ctx_idx, -1))
            for slot, (para, _emb_score) in enumerate(candidates_all):
                ev_text = (para.get("paragraph_text") or "").strip()
                text = f'{prefix_before} Step {hop_j}: {raw_sq} Evidence: "{escape_double_quotes(ev_text)}"'
                texts.append(text)
                owners.append((ctx_idx, slot))

        hiddens_by_layer = batch_last_hidden(
            texts=texts, model=gate_model, tokenizer=gate_tokenizer, layers=[layer],
            batch_size=args.hidden_batch_size, cache=hidden_cache, desc=f"hop{hop_j} hidden",
        )
        hiddens = hiddens_by_layer[layer]

        prefix_hidden: dict[int, np.ndarray] = {}
        cand_hidden: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        for (ctx_idx, slot), h in zip(owners, hiddens):
            if slot == -1:
                prefix_hidden[ctx_idx] = h
            else:
                cand_hidden[ctx_idx][slot] = h

        continuations_by_ex: dict[str, list[tuple[float, float, int, str, str, dict]]] = defaultdict(list)
        for ctx_idx, (ex, path_idx, raw_sq, expanded_q, candidates_all) in enumerate(expand_retrieve):
            h_prev = prefix_hidden[ctx_idx]
            for slot, (para, emb_score) in enumerate(candidates_all):
                if art is not None:
                    gate_score = score_gate_v3(cand_hidden[ctx_idx][slot], h_prev, art)
                else:
                    gate_score = 0.0
                rank_key = emb_score - args.lambda_gate * gate_score
                continuations_by_ex[ex.eid].append((rank_key, gate_score, path_idx, raw_sq, expanded_q, para))

        kept_by_ex: dict[str, list[tuple[float, float, int, str, str, dict]]] = {}
        for ex in examples:
            conts = continuations_by_ex.get(ex.eid)
            if not conts:
                continue
            conts.sort(key=lambda t: t[0], reverse=True)
            kept_by_ex[ex.eid] = conts[: args.beam_width]
            hop_ranked_by_ex[ex.eid].append([int(t[5].get("idx", -1)) for t in conts])

            gold_idx = ex.gold_idxs[hop_j - 1] if hop_j - 1 < len(ex.gold_idxs) else None
            if gold_idx is not None:
                # Full pool (up to --retrieve-k), ranked by rank_key, BEFORE truncating to
                # beam_width -- a real recall@1/@3 over what was actually retrieved+reranked.
                full_ranked = [(para, rk) for rk, _g, _p, _rq, _eq, para in conts]
                full_rank = find_gold_rank(full_ranked, gold_idx)
                full_pool_rank["all"].update(full_rank)
                if ex.K in full_pool_rank:
                    full_pool_rank[ex.K].update(full_rank)

                full_rank_oi = find_gold_rank_set(full_ranked, set(ex.gold_idxs))
                full_pool_rank_orderinvariant["all"].update(full_rank_oi)
                if ex.K in full_pool_rank_orderinvariant:
                    full_pool_rank_orderinvariant[ex.K].update(full_rank_oi)

                survivor_ranked = [(para, rk) for rk, _g, _p, _rq, _eq, para in kept_by_ex[ex.eid]]
                rank = find_gold_rank(survivor_ranked, gold_idx)
                oracle_hop_survive["all"].update(rank)
                if ex.K in oracle_hop_survive:
                    oracle_hop_survive[ex.K].update(rank)

        # Per-hop short-answer generation: still uses hop_steps (expanded) + prior; which
        # prompt builder is selectable via --short-answer-prompt (see SHORT_ANSWER_PROMPT_BUILDERS).
        answer_prompts: list[str] = []
        answer_meta: list[tuple[ExampleState, float, float, int, str, str, dict]] = []
        for ex in examples:
            for rank_key, gate_score, path_idx, raw_sq, expanded_q, para in kept_by_ex.get(ex.eid, []):
                parent = ex.beams[path_idx]
                answer_prompts.append(
                    short_answer_prompt_fn(ex.q_main, parent.hop_steps, parent.prior, expanded_q, para)
                )
                answer_meta.append((ex, rank_key, gate_score, path_idx, raw_sq, expanded_q, para))
        answer_raw = (
            generator.generate_batch(answer_prompts, args.max_new_tokens_answer, desc=f"answer-hop{hop_j}")
            if answer_prompts else []
        )

        rebuilt: dict[str, list[BeamPath]] = defaultdict(list)
        for (ex, rank_key, gate_score, path_idx, raw_sq, expanded_q, para), raw in zip(answer_meta, answer_raw):
            parent = ex.beams[path_idx]
            sub_ans = normalize_short_answer(raw)
            ev_text = (para.get("paragraph_text") or "").strip()
            rebuilt[ex.eid].append(BeamPath(
                hop_steps=parent.hop_steps + [(expanded_q, ev_text)],
                gate_hop_steps=parent.gate_hop_steps + [(raw_sq, ev_text)],
                prior=parent.prior + [sub_ans],
                para_ids=parent.para_ids + [int(para.get("idx", -1))],
                rank_key=rank_key, gate_v3_score=gate_score,
            ))

        for ex in examples:
            if hop_j <= ex.K and ex.eid in rebuilt:
                ex.beams = rebuilt[ex.eid]

        elapsed = time.perf_counter() - t0
        print(f"[hop {hop_j}] {len(texts)} texts scored, elapsed {elapsed / 60:.1f} min", flush=True)

    # Final reader: UNCHANGED -- same prompt/system-prompt as run_retrieval_exp_wavefront_gate_v3.py,
    # reads hop_steps (expanded) + prior, not gate_hop_steps.
    final_tasks: list[tuple[ExampleState, BeamPath]] = []
    for ex in examples:
        if not ex.beams:
            continue
        best = max(ex.beams, key=lambda p: p.rank_key)
        final_tasks.append((ex, best))
    final_reader_prompt_fn = FINAL_READER_PROMPT_BUILDERS[args.final_reader_prompt]
    final_prompts = [
        final_reader_prompt_fn(ex.q_main, best.hop_steps, best.prior) for ex, best in final_tasks
    ]
    final_raw = generator.generate_chat_batch(
        FINAL_READER_SYSTEM_PROMPT, final_prompts, args.max_new_tokens_final, desc="final-reader",
    )

    answer_accum: dict[str, AnswerAccum] = {"all": AnswerAccum(), **{K: AnswerAccum() for K in K_BUCKETS}}
    hop_match_counts: dict[Any, list[int]] = {"all": [0, 0], **{K: [0, 0] for K in K_BUCKETS}}
    chain_match_counts: dict[Any, list[int]] = {"all": [0, 0], **{K: [0, 0] for K in K_BUCKETS}}

    with out_cases.open("w", encoding="utf-8") as f:
        for (ex, best), raw in zip(final_tasks, final_raw):
            fallback = best.prior[-1] if best.prior else ""
            final_answer = parse_final_answer_from_cot(raw, fallback=fallback)
            em, f1 = judge_answer_official(final_answer, ex.row)
            answer_accum["all"].update(em, f1)
            if ex.K in answer_accum:
                answer_accum[ex.K].update(em, f1)

            hop_matches = [pid == gid for pid, gid in zip(best.para_ids, ex.gold_idxs)]
            for key in ("all", ex.K):
                if key in hop_match_counts:
                    hop_match_counts[key][0] += sum(hop_matches)
                    hop_match_counts[key][1] += len(hop_matches)
                    chain_match_counts[key][1] += 1
                    if hop_matches and all(hop_matches):
                        chain_match_counts[key][0] += 1

            f.write(json.dumps({
                "id": ex.eid, "K": ex.K, "question": ex.q_main,
                "final_answer": final_answer, "gold_answer": ex.row.get("answer"),
                "em": em, "f1": f1, "beam_final_rank_key": best.rank_key,
                "beam_final_gate_v3_score": best.gate_v3_score,
                "para_ids": best.para_ids, "gold_idxs": ex.gold_idxs, "prior": best.prior,
                "hop_ranked_para_ids": hop_ranked_by_ex.get(ex.eid, []),
            }, ensure_ascii=False) + "\n")

    def rate(counts: list[int]) -> float | None:
        return round(counts[0] / counts[1], 4) if counts[1] else None

    run_wall_sec = time.perf_counter() - run_t0
    results: dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "output_dir": str(args.out_dir),
        "n_examples": len(final_tasks),
        "timing": {"wall_clock_sec": round(run_wall_sec, 2), "wall_clock_hr": round(run_wall_sec / 3600, 3)},
        "answer_overall": answer_accum["all"].result(),
        "answer_by_K": {K: answer_accum[K].result() for K in K_BUCKETS},
        "final_beam_hop_match_rate_overall": rate(hop_match_counts["all"]),
        "final_beam_hop_match_rate_by_K": {K: rate(hop_match_counts[K]) for K in K_BUCKETS},
        "final_beam_chain_match_rate_overall": rate(chain_match_counts["all"]),
        "final_beam_chain_match_rate_by_K": {K: rate(chain_match_counts[K]) for K in K_BUCKETS},
        "oracle_gold_rank_among_survivors_overall": oracle_hop_survive["all"].result(),
        "oracle_gold_rank_among_survivors_by_K": {K: oracle_hop_survive[K].result() for K in K_BUCKETS},
        "full_pool_gold_rank_overall": full_pool_rank["all"].result(),
        "full_pool_gold_rank_by_K": {K: full_pool_rank[K].result() for K in K_BUCKETS},
        "full_pool_gold_rank_orderinvariant_overall": full_pool_rank_orderinvariant["all"].result(),
        "full_pool_gold_rank_orderinvariant_by_K": {K: full_pool_rank_orderinvariant[K].result() for K in K_BUCKETS},
    }
    out_metrics.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nMetrics saved to  {out_metrics}")
    print(f"Cases saved to    {out_cases}")
    print(f"Wall clock hr     {results['timing']['wall_clock_hr']:.3f}")
    print(f"answer_overall    {results['answer_overall']}")
    print(f"hop_match_rate    {results['final_beam_hop_match_rate_overall']}")
    print(f"chain_match_rate  {results['final_beam_chain_match_rate_overall']}")
    print(f"oracle_survival   {results['oracle_gold_rank_among_survivors_overall']}")
    print(f"full_pool_rank    {results['full_pool_gold_rank_overall']} (real recall@1/@3 "
          f"over the full retrieve-k pool, before beam-width truncation)")
    print(f"  order-invariant {results['full_pool_gold_rank_orderinvariant_overall']} "
          f"(same pool, hit = ANY of this example's gold idxs, not just this hop's -- the "
          f"metric comparable to ChainRAG/GRITHopper's own recall@k)")


if __name__ == "__main__":
    main()
