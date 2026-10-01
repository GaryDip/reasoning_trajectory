#!/usr/bin/env python3
"""
Online, incremental, DAgger-style training of the listwise gate (model.py) -- NOT a
two-phase "collect then train" pipeline. Training happens hop by hop, WITHIN one pass
through the dataset, and the model's own (still-improving) predictions drive what the trace
actually looks like at every subsequent hop:

  hop 1: retrieve real BGE candidates for every example's hop 1. Collect this hop's
         features (see model.py: PCA(query vec), PCA(passage vec), PCA(h_after),
         PCA(delta_h)) + gold label for every example that has gold in its pool. Fit the
         four PCA transforms ONCE on this hop-1 data (never refit again). Train a fresh
         ListwiseGateModel on hop-1's data only. Use this JUST-TRAINED model to score hop
         1's own candidates and commit the argmax pick into every example's trace -- right
         or wrong, whatever the model actually picks is what hop 2 sees, exactly like real
         inference (no gold short-circuiting).
  hop 2: build hop 2's query from hop 1's ACTUAL (possibly wrong) committed short answer.
         Retrieve, collect hop-2 features+labels, ADD them to hop-1's already-collected data
         (not replace it -- avoids the model forgetting what hop 1 taught it). Continue
         training the SAME model (warm-started, not reinitialized) on the hop1+hop2 union.
         Use the updated model to commit hop 2's pick, continue.
  hop 3, 4: same pattern, each time training on everything collected so far.

After the last hop, the model has been incrementally trained across every hop position
using data whose CONTEXT reflects the model's own evolving (sometimes wrong) choices, then
frozen and saved. A clean second pass with the frozen model (run_wavefront_online_gate.py)
is needed for real evaluation, since hop-1 decisions made early in THIS run used a
barely-trained model and aren't representative of the final model's quality.

GT decompose (--decompose-mode gt, the default) is MuSiQue-only, matching how gate v2/v3 were
themselves trained on GT decompose traces, not BART. Per-hop short answers are generated the
same way production does (prompt_short_answer_with_context + vLLM), since the whole point is
that later hops see the model's REAL committed trace, including real (possibly imperfect)
short-answer generation, not a gold-answer shortcut.

--decompose-mode bart_decompose (0831 addition): BART's own sub-questions don't carry GT's
positional guarantee (sub-question i doesn't reliably correspond to gold_idxs[i] -- BART's
hop count/order is independently generated, not copied from the dataset's own annotation), so
gold_local_idx can't be looked up directly. Instead, per hop: take the retrieved candidate
pool's intersection with this example's full gold set (in retrieval-rank order), and ask the
SAME local Llama (already loaded for hidden-state extraction) "can this passage answer this
sub-question, yes or no" for each, in order, stopping at the first yes. If none of them get a
yes, this example is dropped from this hop AND every hop after it for the rest of the run (not
just this one hop) -- a hop with no answerable gold candidate usually means the sub-question
itself is a broken intermediate step (bad BART phrasing, wrong entity), and hops after it are
built on top of that same broken step via [Answer N] expansion, so keeping them would just be
training on more bad labels downstream of the first bad one. This judge-derived target is used
ONLY to pick the softmax label for this hop's training group -- the actual committed
pick/short-answer that propagates to the NEXT hop's query is still the model's own live argmax
over the pool, right or wrong, exactly as in GT mode; DAgger-style self-propagation is
unaffected by this change.

--decompose-mode mixed (0830 addition): pools FOUR sources into one heterogeneous example list
and trains a single model on all of them together, instead of picking one dataset/mode per run:
  1. MuSiQue GT      (positional, free, 100%-reliable -- the "clean floor")
  2. MuSiQue BART    (bart_decompose+judge, same 3000-example K-stratified subset used for the
                      0828 lambda re-tuning work)
  3. 2Wiki BART      (bart_decompose+judge, 3000-example subset, K in {2,4} stratified 1500/1500)
  4. HotpotQA BART   (bart_decompose+judge, 3000-example subset, type bridge/comparison
                      stratified 1500/1500 -- HotpotQA was never in BART's own training mix, so
                      this source's decompose is genuinely out-of-domain for BART and leans on
                      the judge more than the other three)
Rationale: `run_wavefront_online_gate.py` (the real evaluation script) defaults to
--decompose-mode bart_decompose for a reason -- at true inference time there is no gold
decompose to lean on for ANY dataset, MuSiQue included, so a model trained only on MuSiQue-GT
is itself trained on a cleaner input distribution than what it's evaluated on. Mixing in BART's
own (noisier) decompose for all three datasets closes that train/serve gap while still keeping
MuSiQue-GT's free, reliable full-coverage signal as a floor rather than discarding it. Each
ExampleState carries its own .dataset/.decompose_mode (not a single global run-mode), so the
target-judge step below only runs on the bart_decompose-tagged examples within a given hop's
active pool -- MuSiQue-GT examples in that same pool still resolve gold_local_idx positionally,
at zero extra LLM cost. --musique-gt-limit-per-k caps source 1 (default 1000/K, ~3000 total) so
it doesn't numerically dominate the other three ~3000-example sources; --musique-bart-file /
--twowiki-bart-file / --hotpot-bart-file point at sources 2-4's pre-built subset files (regenerate
via decompose/bart/predict.py + convert_predictions_to_v1.py + decompose_to_nl.py if stale).

Usage (small subset first, per the plan -- confirm the loop works before going to full scale):
  python train_online_wavefront.py --split train --limit-per-k 50
  python train_online_wavefront.py --decompose-mode bart_decompose --limit-per-k 50
  python train_online_wavefront.py --decompose-mode mixed --musique-gt-limit-per-k 20 --limit-per-k 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
TRACES_DIR = PROJECT_ROOT / "traces"
sys.path.insert(0, str(TRACES_DIR))
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(HERE))

from run_retrieval_exp import (  # noqa: E402
    K_BUCKETS,
    MUSIQUE_DIR,
    _ST_CACHE,
    _get_pipeline_helpers,
    build_trace_prefix,
    embed_retrieval,
    load_dataset_records,
    load_decompose_index,
    normalize_short_answer,
    resolve_decompose_file,
    select_example_ids,
)
from run_retrieval_exp_wavefront import (  # noqa: E402
    BatchVllmGenerator,
    batch_last_hidden,
    load_gate_model,
    prompt_short_answer_with_context,
    warmup_gate_model_memory,
)
from trace_evidence import gold_evidence_musique  # noqa: E402
from trace_format import escape_double_quotes  # noqa: E402
from model import FeaturePCA, ListwiseGateModel, MLPGateModel, groups_accuracy, listwise_loss, save_artifact  # noqa: E402

DEFAULT_PER_J_LAYERS = {0: 15, 1: 15, 2: 23, 3: 23}


def encode_query_and_docs(query: str, docs: list[str], model_name: str) -> tuple[np.ndarray, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    if model_name not in _ST_CACHE:
        print(f"  [emb] loading {model_name} …", file=sys.stderr, flush=True)
        _ST_CACHE[model_name] = SentenceTransformer(model_name)
    st = _ST_CACHE[model_name]
    qv = st.encode([query], normalize_embeddings=True)[0]
    dv = st.encode(docs, normalize_embeddings=True) if docs else np.zeros((0, qv.shape[0]))
    return qv, dv


def build_target_judge_prompt(sub_question: str, evidence_text: str) -> str:
    """bart_decompose mode only: ask whether a gold passage actually answers a given
    (possibly mis-phrased) BART sub-question, before trusting it as this hop's training
    target. Kept deliberately terse (single yes/no word) -- this runs once per (example,
    gold-in-pool candidate) per hop, not something to spend many tokens on."""
    return (
        f'Passage: "{escape_double_quotes(evidence_text)}"\n'
        f"Question: {sub_question}\n"
        "Can this passage answer the question? Reply with exactly one word: Yes or No."
    )


def parse_yes_no(raw: str) -> bool:
    return (raw or "").strip().lower().startswith("y")


@dataclass
class BeamPath:
    hop_steps: list[tuple[str, str]] = field(default_factory=list)
    prior: list[str] = field(default_factory=list)
    para_ids: list[int] = field(default_factory=list)


@dataclass
class ExampleState:
    eid: str  # globally unique across sources: "{dataset}:{decompose_mode}:{raw_eid}"
    raw_eid: str  # the underlying dataset's own id (not unique across sources by itself --
    # e.g. the same MuSiQue question can appear once via the GT source and once via the
    # MuSiQue-BART source in --decompose-mode mixed, as two independent ExampleStates)
    dataset: str  # "musique" | "2wiki" | "hotpot"
    decompose_mode: str  # "gt" | "bart_decompose" -- per-example, not a single global run mode,
    # so --decompose-mode mixed can pool sources that each resolve gold_local_idx differently
    row: dict[str, Any]
    q_main: str
    paragraphs: list[dict[str, Any]]
    sub_questions: list[str]
    gold_idxs: list[int]
    K: int
    beam: BeamPath
    # bart_decompose examples only: set True the first hop where no gold-in-pool candidate
    # passes the "can this answer the sub-question" judge -- excluded from every hop from then
    # on (see module docstring). Always False for gt examples.
    broken: bool = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique"], default="musique",
                     help="GT decompose only exists for MuSiQue -- this training procedure "
                          "cannot run on 2wiki/hotpot, same constraint gate v2/v3 have. "
                          "--decompose-mode bart_decompose still only reads MuSiQue's own BART "
                          "predictions here; extending to 2wiki/hotpot needs their own BART "
                          "train-split predictions generated first (only dev exists today). "
                          "Ignored under --decompose-mode mixed, which always pools all three "
                          "datasets regardless of this flag.")
    ap.add_argument("--decompose-mode", choices=["gt", "bart_decompose", "mixed"], default="gt",
                     help="gt (default): gold_local_idx comes directly from "
                          "gold_idxs[hop_j-1] (MuSiQue's own annotated hop order), no extra "
                          "LLM calls -- unchanged original behavior. bart_decompose: BART's "
                          "own sub-questions don't have that positional guarantee, so "
                          "gold_local_idx is instead determined per hop by --target-verify-* "
                          "below (retrieve, then ask the same local Llama 'can this gold "
                          "passage answer this sub-question', in retrieval-rank order, first "
                          "yes wins; no yes at all -> this example is dropped from this hop "
                          "AND every later hop, see module docstring addendum). mixed: pools "
                          "MuSiQue-GT + MuSiQue-BART + 2Wiki-BART + HotpotQA-BART into one "
                          "example list and trains a single model on all four together, see "
                          "module docstring addendum.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--twowiki-file", type=Path, default=None,
                     help="Raw 2Wiki json, used for --dataset 2wiki (not currently offered "
                          "outside mixed mode) and --decompose-mode mixed's 2Wiki-BART source. "
                          "Defaults to data/raw/2wikimultihopqa/{train,dev}.json matching --split.")
    ap.add_argument("--hotpot-file", type=Path, default=None,
                     help="Raw HotpotQA json, same role as --twowiki-file. Defaults to "
                          "data/raw/hotpotqa/{hotpot_train_v1.1,hotpot_dev_distractor_v1}.json "
                          "matching --split.")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--musique-bart-file", type=Path,
                     default=PROJECT_ROOT / "decompose/bart/data/musique_train_lambda_subset_big_bart_nl.jsonl",
                     help="mixed mode only: MuSiQue-BART source (3000-example K-stratified "
                          "subset, same file the 0828 lambda re-tuning sweep used).")
    ap.add_argument("--twowiki-bart-file", type=Path,
                     default=PROJECT_ROOT / "decompose/bart/data/2wiki_train_online_gate_subset3k_bart_nl.jsonl",
                     help="mixed mode only: 2Wiki-BART source (3000-example subset, K in {2,4} "
                          "stratified 1500/1500, generated 0830 -- see decompose/bart/predict.py "
                          "+ convert_predictions_to_v1.py + decompose_to_nl.py to regenerate).")
    ap.add_argument("--hotpot-bart-file", type=Path,
                     default=PROJECT_ROOT / "decompose/bart/data/hotpot_train_online_gate_subset3k_bart_nl.jsonl",
                     help="mixed mode only: HotpotQA-BART source (3000-example subset, type "
                          "bridge/comparison stratified 1500/1500, generated 0830).")
    ap.add_argument("--musique-gt-limit-per-k", type=int, default=1000,
                     help="mixed mode only: cap on the MuSiQue-GT source, stratified by K "
                          "(default 1000/K =~ 3000 total, matching the other three sources' "
                          "~3000-example scale so GT doesn't numerically dominate the pooled "
                          "loss). 0 = no cap (full MuSiQue train, ~10x+ the other sources).")
    ap.add_argument("--target-verify-max-new-tokens", type=int, default=4,
                     help="bart_decompose mode only: max tokens for the yes/no judge call.")
    ap.add_argument("--enhance-file", type=Path, default=None,
                     help="Extra K=3/4 paraphrased GT decompositions (traces/enhance_decompose_nl.py "
                          "output, e.g. train_nl_enhance.jsonl) to add ON TOP OF --decompose-file, to "
                          "counter K=2's ~72%% dominance in raw MuSiQue train. Each row's own 'id' "
                          "(e.g. '..._enh0') is distinct from its 'source_id' (the raw record it reuses "
                          "paragraphs/gold evidence from) -- only rows are merged, no synthetic Llama "
                          "answers or reader text, no new retrieval targets. Auto-detected as "
                          "'<decompose-file stem>_enhance.jsonl' next to --decompose-file if present and "
                          "--split train; pass an explicit path to override, or a nonexistent path has "
                          "no effect (falls back to --decompose-file alone). Only verify_pass=true rows "
                          "are used.")
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--answerable-only", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--limit-per-k", type=int, default=0,
                     help="Stratified-by-K subset, for validating the training loop before "
                          "committing to a full run (see module docstring).")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--model-type", choices=["bilinear", "mlp"], default="bilinear",
                     help="bilinear (default): ListwiseGateModel, two bilinear interaction "
                          "terms + a learned linear mixing gate, no MLP/activation -- the "
                          "production design (see model.py docstring for why). mlp: "
                          "MLPGateModel, a 2-hidden-layer ReLU MLP over the same concatenated "
                          "features -- an ABLATION to check whether the bilinear model's "
                          "restricted capacity (not just --pca-dim) is leaving accuracy on the "
                          "table, not a proposed replacement for the production gate.")
    ap.add_argument("--mlp-hidden-dim", type=int, default=128,
                     help="--model-type mlp only: hidden layer width.")
    ap.add_argument("--include-h-prev", action="store_true", default=False,
                     help="Add PCA(h_prev) as a 5th feature branch (on top of q/d/h_after/delta) "
                          "-- gives the model direct access to 'what was the state right before "
                          "this pick', not just the single local delta=h_after-h_prev. Requires "
                          "--model-type mlp (ListwiseGateModel's bilinear pairing has no slot "
                          "for an unpaired 5th branch).")
    ap.add_argument("--cache-features-dir", type=Path, default=None,
                     help="Dump this run's raw (pre-PCA) per-hop feature arrays + gold_local "
                          "labels + per-example (eid/dataset/decompose_mode) metadata to this "
                          "directory as a side effect of the normal run -- retrieval, hidden-"
                          "state extraction, target-judge, and short-answer generation stay "
                          "exactly as they are, this just additionally persists their output. "
                          "Lets refit_from_cache.py re-run ONLY the cheap part (PCA fit + model "
                          "training, seconds-to-minutes) for a --pca-dim/--model-type/"
                          "--include-h-prev sweep, instead of repeating the expensive part "
                          "(retrieval + Llama forward passes + judge LLM calls, hours) for every "
                          "point in the sweep. Caveat: hop 2+'s retrieval/hidden-states depend on "
                          "THIS run's own committed picks (ex.beam.prior, DAgger-style) -- a "
                          "sweep refit from one cached run compares architectures/dims on one "
                          "FIXED trajectory, not each architecture's own from-scratch online "
                          "trajectory. Fine for screening; re-run this script directly (no cache) "
                          "for a true end-to-end confirmation of whatever wins the sweep.")
    ap.add_argument("--val-frac", type=float, default=0.1,
                     help="Each hop independently holds out this fraction of ITS OWN new "
                          "trainable groups as validation (added to the accumulated validation "
                          "pool, same as the training pool); NOT a single up-front split, so the "
                          "same example can be train at one hop and val at another.")
    ap.add_argument("--patience", type=int, default=5,
                     help="Stop this hop's training once accumulated val loss hasn't improved "
                          "for this many consecutive epochs; the best-val-loss checkpoint (not "
                          "the last epoch) is what's kept.")
    ap.add_argument("--max-epochs-per-hop", type=int, default=200,
                     help="Safety cap on epochs per hop -- early stopping should trigger well "
                          "before this on a healthy run; replaces the old fixed "
                          "--train-epochs-per-hop (pool size varies 24x across hops 1->4, a "
                          "fixed epoch count under/over-trains most of them).")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--results-root", type=Path, default=HERE / "results")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--out-artifact", type=Path, default=None)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--gate-device", default="cuda:0")
    ap.add_argument("--hidden-batch-size", type=int, default=8)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens-answer", type=int, default=64)
    return ap.parse_args()


def load_source_examples(
    *,
    dataset: str,
    decompose_mode: str,
    decompose_file: Path,
    musique_dir: Path,
    twowiki_file: Path | None,
    hotpot_file: Path | None,
    split: str,
    answerable_only: bool,
    limit: int,
    limit_per_k: int,
    sample_seed: int,
    enhance_file: Path | None,
    allow_enhance_autodetect: bool = True,
) -> list[ExampleState]:
    """Load one (dataset, decompose_mode) source's examples, tagged with .dataset/
    .decompose_mode/.raw_eid, eid made globally-unique as "{dataset}:{decompose_mode}:{raw_eid}".
    Shared by both the single-source CLI modes (gt/bart_decompose) and --decompose-mode mixed
    (which calls this once per of its four sources) -- this is exactly the loading logic that
    used to live inline in main() before mixed mode existed, unchanged in behavior for the
    single-source case."""
    ds_args = argparse.Namespace(
        dataset=dataset, musique_dir=musique_dir, split=split,
        twowiki_file=twowiki_file, hotpot_file=hotpot_file,
    )
    records = load_dataset_records(ds_args)
    decompose_idx = load_decompose_index(decompose_file)
    # record_id_of[eid]: which raw-record id to pull paragraphs/gold-evidence/answerable from.
    # For the primary decompose file this is always eid itself; enhance rows below override it
    # to their source_id, since an enhance row's own 'id' (e.g. '..._enh0') never appears in the
    # raw records -- only its paraphrased sub_questions are new, the underlying question's
    # paragraphs/answer/gold evidence are unchanged. (enhance is GT-only/MuSiQue-only in practice.)
    record_id_of: dict[str, str] = {eid: eid for eid in decompose_idx}
    # question_of[eid]: overrides row["question"] for enhance rows -- the enhance script
    # paraphrases the main question together with its sub_questions as one coherent unit, so an
    # enhance row's chain must be read against ITS OWN paraphrased question, not the raw
    # record's original phrasing (entities/wording can differ between the two paraphrases).
    question_of: dict[str, str] = {}

    if enhance_file is None and allow_enhance_autodetect and split == "train":
        candidate = decompose_file.with_name(f"{decompose_file.stem}_enhance{decompose_file.suffix}")
        enhance_file = candidate if candidate.exists() else None
    if enhance_file and enhance_file.exists():
        n_seen = n_kept = 0
        with enhance_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                n_seen += 1
                if not obj.get("verify_pass"):
                    continue
                eid = str(obj.get("id", "")).strip()
                source_id = str(obj.get("source_id", "")).strip()
                sqs = obj.get("sub_questions") or []
                question = str(obj.get("question", "")).strip()
                if not eid or not source_id or not isinstance(sqs, list) or not sqs or not question:
                    continue
                decompose_idx[eid] = [str(s) for s in sqs if str(s).strip()]
                record_id_of[eid] = source_id
                question_of[eid] = question
                n_kept += 1
        print(f"  [{dataset}/{decompose_mode}] enhance file: {enhance_file} ({n_kept}/{n_seen} rows kept, verify_pass=true)")

    ids = sorted(eid for eid in decompose_idx if record_id_of.get(eid, eid) in records)
    if answerable_only:
        ids = [eid for eid in ids if records[record_id_of[eid]].get("answerable", True)]
    ids = select_example_ids(ids, decompose_idx, limit=limit, limit_per_k=limit_per_k, seed=sample_seed)
    # select_example_ids only filters out K not in K_BUCKETS when limit_per_k > 0 (its
    # stratified-sampling path) -- limit_per_k == 0 ("use everything") skips that check entirely,
    # so BART's own occasional over/under-decomposition (a handful of predicted sub_questions
    # lists longer than BART_MAX_HOPS=4, e.g. 3 K=5 / 1 K=6 example in the HotpotQA-BART subset)
    # can otherwise leak through uncapped and inflate max_hops for the WHOLE run. gold_idxs is
    # always capped at BART_MAX_HOPS=4 regardless (gold_evidence_musique/normalize_open_qa_record),
    # so any hop beyond 4 would never get a gold label anyway -- just wasted retrieval + hidden-
    # state extraction for a few stray examples. Filter here unconditionally, not only when
    # limit_per_k > 0.
    n_before = len(ids)
    ids = [eid for eid in ids if len(decompose_idx.get(eid) or []) in K_BUCKETS]
    if len(ids) != n_before:
        print(f"  [{dataset}/{decompose_mode}] dropped {n_before - len(ids)} example(s) with "
              f"predicted hop count outside K_BUCKETS={K_BUCKETS}")

    out: list[ExampleState] = []
    for raw_eid in ids:
        row = records[record_id_of[raw_eid]]
        sub_qs = decompose_idx.get(raw_eid) or []
        if not sub_qs:
            continue
        _gold_texts, gold_idxs = gold_evidence_musique(row)  # dataset-agnostic despite the name
        q_main = question_of.get(raw_eid) or (row.get("question") or "").strip()
        out.append(ExampleState(
            eid=f"{dataset}:{decompose_mode}:{raw_eid}", raw_eid=raw_eid,
            dataset=dataset, decompose_mode=decompose_mode,
            row=row, q_main=q_main,
            paragraphs=row.get("paragraphs") or [], sub_questions=sub_qs,
            gold_idxs=gold_idxs, K=len(sub_qs), beam=BeamPath(),
        ))
    return out


def main() -> None:
    run_t0 = time.perf_counter()
    args = parse_args()
    if args.include_h_prev and args.model_type != "mlp":
        raise SystemExit("--include-h-prev requires --model-type mlp (ListwiseGateModel's "
                          "bilinear pairing has no slot for an unpaired 5th branch)")
    if args.twowiki_file is None:
        args.twowiki_file = PROJECT_ROOT / "data/raw/2wikimultihopqa" / ("train.json" if args.split == "train" else "dev.json")
    if args.hotpot_file is None:
        args.hotpot_file = PROJECT_ROOT / "data/raw/hotpotqa" / (
            "hotpot_train_v1.1.json" if args.split == "train" else "hotpot_dev_distractor_v1.json")
    if args.decompose_mode != "mixed":
        args.decompose_file = resolve_decompose_file(args.decompose_mode, args.split, args.decompose_file, args.dataset)

    tag = args.run_tag or f"online_listwise_{args.split}"
    out_dir = args.out_dir or (args.results_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_artifact = args.out_artifact or (out_dir / "artifact.joblib")
    print(f"Output directory: {out_dir}")

    (expand_hop_template, *_rest, _judge) = _get_pipeline_helpers()

    source_summary: list[dict[str, Any]] = []
    if args.decompose_mode == "mixed":
        musique_gt_file = resolve_decompose_file("gt", args.split, None, "musique")
        # --limit-per-k (the pre-existing generic flag) additionally caps the three
        # already-~3000-example BART sources here -- 0 (default) leaves them uncapped (use the
        # whole pre-built subset file); set it for a quick smoke test of --decompose-mode mixed
        # without waiting on all four sources at full scale.
        mixed_sources = [
            dict(dataset="musique", decompose_mode="gt", decompose_file=musique_gt_file,
                 limit_per_k=args.musique_gt_limit_per_k, use_enhance=True),
            dict(dataset="musique", decompose_mode="bart_decompose", decompose_file=args.musique_bart_file,
                 limit_per_k=args.limit_per_k, use_enhance=False),
            dict(dataset="2wiki", decompose_mode="bart_decompose", decompose_file=args.twowiki_bart_file,
                 limit_per_k=args.limit_per_k, use_enhance=False),
            dict(dataset="hotpot", decompose_mode="bart_decompose", decompose_file=args.hotpot_bart_file,
                 limit_per_k=args.limit_per_k, use_enhance=False),
        ]
        examples = []
        for src in mixed_sources:
            src_examples = load_source_examples(
                dataset=src["dataset"], decompose_mode=src["decompose_mode"],
                decompose_file=src["decompose_file"], musique_dir=args.musique_dir,
                twowiki_file=args.twowiki_file, hotpot_file=args.hotpot_file, split=args.split,
                answerable_only=args.answerable_only, limit=0, limit_per_k=src["limit_per_k"],
                sample_seed=args.sample_seed,
                enhance_file=(args.enhance_file if src["use_enhance"] else None),
                allow_enhance_autodetect=src["use_enhance"],
            )
            print(f"  [{src['dataset']}/{src['decompose_mode']}] {len(src_examples)} examples "
                  f"(decompose={src['decompose_file']})")
            source_summary.append({"dataset": src["dataset"], "decompose_mode": src["decompose_mode"],
                                    "decompose_file": str(src["decompose_file"]), "n_examples": len(src_examples)})
            examples.extend(src_examples)
        print(f"Examples (mixed pool, 4 sources): {len(examples)} total")
    else:
        examples = load_source_examples(
            dataset=args.dataset, decompose_mode=args.decompose_mode, decompose_file=args.decompose_file,
            musique_dir=args.musique_dir, twowiki_file=args.twowiki_file, hotpot_file=args.hotpot_file,
            split=args.split, answerable_only=args.answerable_only, limit=args.limit,
            limit_per_k=args.limit_per_k, sample_seed=args.sample_seed, enhance_file=args.enhance_file,
        )
        source_summary.append({"dataset": args.dataset, "decompose_mode": args.decompose_mode,
                                "decompose_file": str(args.decompose_file), "n_examples": len(examples)})
        print(f"Examples: {len(examples)}  decompose({args.decompose_mode})={args.decompose_file}")

    print("Loading Llama model for hidden-state features ...", flush=True)
    gate_model, gate_tokenizer = load_gate_model(args)
    warmup_gate_model_memory(gate_model, gate_tokenizer, args.hidden_batch_size)
    print("Loading vLLM generator (for real per-hop short answers) ...", flush=True)
    generator = BatchVllmGenerator(
        args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
    )

    hidden_cache: dict[tuple[int, str], Any] = {}
    feature_pca: FeaturePCA | None = None
    model: ListwiseGateModel | MLPGateModel | None = None
    optimizer: torch.optim.Optimizer | None = None
    # Split happens fresh at EACH hop, independently, from that hop's own new groups
    # (--val-frac of hop_j's own examples -> val, rest -> train) -- not a single up-front
    # example-level split -- so the same example can land in train at one hop and val at
    # another; that's intentional, each hop's groups are their own data point.
    #
    # accumulated_train_groups: flat, pooled across all hops so far -- training itself SHOULD
    # treat hop1..hop_j as one undifferentiated pool (that's the whole incremental-training
    # design, see module docstring).
    #
    # val_groups_by_hop: kept SEPARATE per hop (not flattened) on purpose. Early stopping needs
    # to know whether THIS hop's new data has been learned, not just whether the accumulated
    # pool's overall loss looks fine -- a flat pooled validation set would let hop 1/2's much
    # larger, already-well-fit validation counts drown out a newer hop's still-poor fit (this
    # is exactly what caused hop 3/4 to stop after only 6-8 epochs in an earlier run: the
    # pooled val loss barely moved because hop1+hop2 dominated it numerically, even though
    # hop3's own fit was still bad). Averaging equally-weighted PER-HOP metrics instead means
    # a newer, smaller hop's validation performance counts just as much as an older, bigger
    # one's when deciding whether to keep training.
    accumulated_train_groups: list[tuple[torch.Tensor, int]] = []
    val_groups_by_hop: dict[int, list[tuple[torch.Tensor, int]]] = {}
    device = torch.device("cpu")  # PCA'd features are small (4*64=256-dim); CPU is plenty

    max_hops = max((ex.K for ex in examples), default=0)
    per_hop_stats: list[dict] = []

    for hop_j in range(1, max_hops + 1):
        t0 = time.perf_counter()
        print(f"\n=== hop {hop_j}/{max_hops} ===", flush=True)
        layer = DEFAULT_PER_J_LAYERS.get(hop_j - 1, 23)

        active = [ex for ex in examples if hop_j <= ex.K and not ex.broken]
        if not active:
            # Defensive: shouldn't normally happen (the K-filter above keeps max_hops aligned
            # with real data), but if every example eligible for this hop got marked .broken at
            # an earlier hop, there is nothing to retrieve/extract features for -- skip cleanly
            # rather than calling feature_pca.transform() on a 0-row array, which sklearn's PCA
            # rejects outright (crashes the whole run, losing every hop trained so far, since
            # save_artifact() only runs after the loop completes normally).
            print(f"[hop {hop_j}] 0 examples active (all broken or excluded) -- skipping this hop", flush=True)
            per_hop_stats.append({"hop": hop_j, "n_active": 0, "skipped": True})
            continue
        raw_sqs = [ex.sub_questions[hop_j - 1] for ex in active]
        expanded_qs = [expand_hop_template(sq, ex.beam.prior) for sq, ex in zip(raw_sqs, active)]

        # Retrieval: real BGE top-k pool, unchanged from how production retrieves -- this
        # model only reranks WITHIN that pool, it doesn't change what gets retrieved.
        candidates_per_ex: list[list[dict]] = []
        q_vecs: list[np.ndarray] = []
        d_vecs_per_ex: list[np.ndarray] = []
        for ex, expanded_q in zip(active, expanded_qs):
            ranked = embed_retrieval(expanded_q, ex.paragraphs, args.retrieve_k, args.cos_model)
            paras = [p for p, _s in ranked]
            candidates_per_ex.append(paras)
            docs = [f"{p.get('title') or ''}\n{(p.get('paragraph_text') or '').strip()}" for p in paras]
            qv, dv = encode_query_and_docs(expanded_q, docs, args.cos_model)
            q_vecs.append(qv)
            d_vecs_per_ex.append(dv)
        print(f"[hop {hop_j}] {len(active)} examples retrieved", flush=True)

        # Determine this hop's gold_local_idx for EVERY active example, BEFORE the expensive
        # hidden-state extraction below (so bart_decompose examples newly broken here get
        # excluded from it too -- saves real compute at full-dataset scale, not just training-
        # group construction). Per-example branch on .decompose_mode, not a single global run
        # mode, so --decompose-mode mixed can pool gt and bart_decompose examples in the same
        # hop's active batch: gt examples resolve positionally (gold_idxs[hop_j-1], free, no
        # LLM calls); bart_decompose examples go through the retrieval-rank + LLM judge below.
        # See module docstring for both.
        gold_local_by_i: dict[int, int | None] = {}
        bart_positions = [i for i, ex in enumerate(active) if ex.decompose_mode == "bart_decompose"]
        if bart_positions:
            judge_prompts: list[str] = []
            judge_owner: list[tuple[int, int]] = []  # (i, candidate slot)
            gold_in_pool_by_i: dict[int, list[int]] = {}  # i -> gold-in-pool slots, retrieval-rank order
            for i in bart_positions:
                ex, paras = active[i], candidates_per_ex[i]
                gold_set = set(ex.gold_idxs)
                gold_slots = [s for s, p in enumerate(paras) if int(p.get("idx", -1)) in gold_set]
                gold_in_pool_by_i[i] = gold_slots
                for slot in gold_slots:
                    ev_text = (paras[slot].get("paragraph_text") or "").strip()
                    judge_prompts.append(build_target_judge_prompt(raw_sqs[i], ev_text))
                    judge_owner.append((i, slot))

            judge_raw = (
                generator.generate_batch(judge_prompts, args.target_verify_max_new_tokens,
                                          desc=f"hop{hop_j} target-judge")
                if judge_prompts else []
            )
            judge_yes = {owner: parse_yes_no(raw) for owner, raw in zip(judge_owner, judge_raw)}

            # Log every (example, gold-in-pool candidate) judge call for spot-checking --
            # verdicts are only ever used internally otherwise, no way to audit them after the
            # fact without this.
            with (out_dir / "target_judge_log.jsonl").open("a", encoding="utf-8") as jf:
                for (i, slot), raw in zip(judge_owner, judge_raw):
                    ex = active[i]
                    para = candidates_per_ex[i][slot]
                    jf.write(json.dumps({
                        "hop": hop_j, "eid": ex.eid, "dataset": ex.dataset,
                        "sub_question": raw_sqs[i],
                        "evidence_title": para.get("title"),
                        "evidence_text": (para.get("paragraph_text") or "").strip(),
                        "raw_response": raw, "verdict_yes": parse_yes_no(raw),
                    }, ensure_ascii=False) + "\n")

            newly_broken: set[int] = set()
            for i in bart_positions:
                chosen = next((s for s in gold_in_pool_by_i.get(i, []) if judge_yes.get((i, s))), None)
                if chosen is None:
                    active[i].broken = True
                    newly_broken.add(i)
                else:
                    gold_local_by_i[i] = chosen
            if newly_broken:
                print(f"[hop {hop_j}] {len(newly_broken)} example(s) newly broken (no gold-in-pool "
                      f"candidate answered its sub-question) -- dropped from this hop and all later "
                      f"hops", flush=True)

            if newly_broken:
                keep_idx = [i for i in range(len(active)) if i not in newly_broken]
                active = [active[i] for i in keep_idx]
                raw_sqs = [raw_sqs[i] for i in keep_idx]
                expanded_qs = [expanded_qs[i] for i in keep_idx]
                candidates_per_ex = [candidates_per_ex[i] for i in keep_idx]
                q_vecs = [q_vecs[i] for i in keep_idx]
                d_vecs_per_ex = [d_vecs_per_ex[i] for i in keep_idx]
                gold_local_by_i = {new_i: gold_local_by_i[old_i] for new_i, old_i in enumerate(keep_idx)
                                    if old_i in gold_local_by_i}

        # gt examples (untouched by the judge block above): resolve gold_local_idx positionally.
        for i, ex in enumerate(active):
            if ex.decompose_mode == "gt":
                paras = candidates_per_ex[i]
                gold_idx = ex.gold_idxs[hop_j - 1] if hop_j - 1 < len(ex.gold_idxs) else None
                gold_local_by_i[i] = (
                    next((s for s, p in enumerate(paras) if int(p.get("idx", -1)) == gold_idx), None)
                    if gold_idx is not None else None
                )

        # Gate-scoring hidden states: prefix_before (no candidate) + one text per candidate,
        # same text format gate v2/v3 train on.
        texts: list[str] = []
        owners: list[tuple[int, int]] = []
        for i, (ex, raw_sq, expanded_q, paras) in enumerate(zip(active, raw_sqs, expanded_qs, candidates_per_ex)):
            prefix_before = build_trace_prefix(ex.q_main, ex.beam.hop_steps)
            texts.append(prefix_before)
            owners.append((i, -1))
            for slot, para in enumerate(paras):
                ev_text = (para.get("paragraph_text") or "").strip()
                text = f'{prefix_before} Step {hop_j}: {expanded_q} Evidence: "{escape_double_quotes(ev_text)}"'
                texts.append(text)
                owners.append((i, slot))

        hiddens_by_layer = batch_last_hidden(
            texts=texts, model=gate_model, tokenizer=gate_tokenizer, layers=[layer],
            batch_size=args.hidden_batch_size, cache=hidden_cache, desc=f"hop{hop_j} hidden",
        )
        hiddens = hiddens_by_layer[layer]
        h_prev_by_ex: dict[int, np.ndarray] = {}
        h_cand_by_ex: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        for (i, slot), h in zip(owners, hiddens):
            if slot == -1:
                h_prev_by_ex[i] = h
            else:
                h_cand_by_ex[i][slot] = h

        # Build this hop's raw (pre-PCA) feature arrays + gold-in-pool labels.
        hop_q, hop_d, hop_h, hop_delta, hop_hprev = [], [], [], [], []
        hop_group_bounds: list[tuple[int, int, int | None]] = []  # (start, end, gold_local_idx_or_None)
        cursor = 0
        for i, (ex, paras) in enumerate(zip(active, candidates_per_ex)):
            n = len(paras)
            if n == 0:
                hop_group_bounds.append((cursor, cursor, None))
                continue
            # gold_local_by_i is populated for every i above (gt: positional; bart_decompose:
            # judge-derived), regardless of this example's mode.
            gold_local = gold_local_by_i.get(i)
            h_prev = h_prev_by_ex[i]
            for slot in range(n):
                hop_q.append(q_vecs[i])
                hop_d.append(d_vecs_per_ex[i][slot])
                h_after = h_cand_by_ex[i][slot]
                hop_h.append(h_after)
                hop_delta.append(h_after - h_prev)
                if args.include_h_prev:
                    hop_hprev.append(h_prev)  # same h_prev repeated per candidate slot -- it
                    # doesn't depend on which candidate, only on the state before this hop
            hop_group_bounds.append((cursor, cursor + n, gold_local))
            cursor += n

        hop_q_arr = np.stack(hop_q, axis=0) if hop_q else np.zeros((0, 768))
        hop_d_arr = np.stack(hop_d, axis=0) if hop_d else np.zeros((0, 768))
        hop_h_arr = np.stack(hop_h, axis=0) if hop_h else np.zeros((0, 4096))
        hop_delta_arr = np.stack(hop_delta, axis=0) if hop_delta else np.zeros((0, 4096))
        hop_hprev_arr = (np.stack(hop_hprev, axis=0) if hop_hprev else np.zeros((0, 4096))) if args.include_h_prev else None

        if args.cache_features_dir:
            args.cache_features_dir.mkdir(parents=True, exist_ok=True)
            starts = np.array([b[0] for b in hop_group_bounds], dtype=np.int64)
            ends = np.array([b[1] for b in hop_group_bounds], dtype=np.int64)
            golds = np.array([b[2] if b[2] is not None else -1 for b in hop_group_bounds], dtype=np.int64)
            np.savez_compressed(
                args.cache_features_dir / f"hop{hop_j}.npz",
                q=hop_q_arr.astype(np.float32), d=hop_d_arr.astype(np.float32),
                # h/delta/hprev are 4096-dim Llama hidden states -- float16 halves the cache
                # size; they get PCA'd down to pca_dim (<=128 in every sweep point we care
                # about) right after loading, so this precision loss is negligible for that.
                h=hop_h_arr.astype(np.float16), delta=hop_delta_arr.astype(np.float16),
                hprev=(hop_hprev_arr.astype(np.float16) if hop_hprev_arr is not None else np.zeros((0, 0), dtype=np.float16)),
                starts=starts, ends=ends, golds=golds,
            )
            (args.cache_features_dir / f"hop{hop_j}_meta.json").write_text(json.dumps([
                {"eid": ex.eid, "dataset": ex.dataset, "decompose_mode": ex.decompose_mode}
                for ex in active
            ]), encoding="utf-8")
            print(f"[hop {hop_j}] cached raw features -> {args.cache_features_dir / f'hop{hop_j}.npz'}", flush=True)

        if feature_pca is None:
            n_branches = 5 if args.include_h_prev else 4
            print(f"[hop {hop_j}] fitting the {n_branches} PCA transforms on hop-1 data ({len(hop_q)} candidates) ...")
            feature_pca = FeaturePCA(pca_dim=args.pca_dim, include_h_prev=args.include_h_prev)
            feature_pca.fit(hop_q_arr, hop_d_arr, hop_h_arr, hop_delta_arr, hop_hprev_arr)
            if args.model_type == "mlp":
                model = MLPGateModel(feature_pca.out_dim, hidden_dim=args.mlp_hidden_dim)
            else:
                model = ListwiseGateModel(*feature_pca.branch_dims)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        transformed = feature_pca.transform(hop_q_arr, hop_d_arr, hop_h_arr, hop_delta_arr, hop_hprev_arr)
        transformed_t = torch.tensor(transformed, dtype=torch.float32, device=device)

        new_groups: list[tuple[torch.Tensor, int]] = []
        group_feats_by_ex: dict[int, torch.Tensor] = {}
        for i, (start, end, gold_local) in enumerate(hop_group_bounds):
            feats = transformed_t[start:end]
            group_feats_by_ex[i] = feats
            if gold_local is not None and end > start:
                new_groups.append((feats, gold_local))

        # This hop's own --val-frac split, independent of every other hop's split.
        rng = random.Random(args.sample_seed * 1000 + hop_j)
        shuffled = new_groups[:]
        rng.shuffle(shuffled)
        n_val = round(len(shuffled) * args.val_frac)
        new_val_groups, new_train_groups = shuffled[:n_val], shuffled[n_val:]
        accumulated_train_groups.extend(new_train_groups)
        val_groups_by_hop[hop_j] = new_val_groups
        n_val_total = sum(len(g) for g in val_groups_by_hop.values())
        print(f"[hop {hop_j}] +{len(new_train_groups)} train / +{len(new_val_groups)} val groups "
              f"(gold present in pool), {len(accumulated_train_groups)}/{n_val_total} "
              f"accumulated total", flush=True)

        # Incremental training: warm-started, on the FULL accumulated train pool (hop 1..hop_j),
        # not just this hop's new data -- see module docstring. Epoch count is no longer fixed --
        # early-stop once the model stops improving, instead of always running the same number
        # of steps regardless of how big the pool is.
        #
        # Stopping metric: mean of PICK ACCURACY (not loss) computed SEPARATELY per hop, then
        # averaged across hops with equal weight -- not loss, and not one flat pooled
        # computation. Two changes from the first version of this loop, both because loss
        # dropped smoothly even while pick accuracy on newer hops was still bad, and because a
        # flat pooled computation (of either metric) lets older/bigger hops' already-good
        # numbers hide a newer hop's still-poor fit. Accuracy also matches the metric this
        # script actually cares about (does the model pick the right candidate), which loss is
        # only a smooth proxy for.
        best_val_metric = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        epochs_since_improve = 0
        epochs_run = 0
        if accumulated_train_groups:
            model.train()
            for epoch in range(args.max_epochs_per_hop):
                optimizer.zero_grad()
                loss = listwise_loss(model, accumulated_train_groups)
                loss.backward()
                optimizer.step()
                epochs_run = epoch + 1

                model.eval()
                hop_accs = [groups_accuracy(model, g) for g in val_groups_by_hop.values() if g]
                val_metric = sum(hop_accs) / len(hop_accs) if hop_accs else groups_accuracy(model, accumulated_train_groups)
                model.train()

                if val_metric > best_val_metric + 1e-4:
                    best_val_metric = val_metric
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= args.patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            print(f"[hop {hop_j}] trained {epochs_run} epochs (early-stopped, patience="
                  f"{args.patience}) on {len(accumulated_train_groups)} train groups, "
                  f"best mean-per-hop val_accuracy={best_val_metric:.4f} over {len(val_groups_by_hop)} "
                  f"hops' validation sets ({n_val_total} groups total)", flush=True)

        # Use the just-updated model to make the REAL pick for this hop, right or wrong.
        model.eval()
        n_correct = 0
        # Per-(dataset, decompose_mode) breakdown of the same accuracy -- --decompose-mode mixed
        # pools four very different-quality sources (100%-reliable positional GT labels vs.
        # judge-derived BART labels of varying reliability per dataset) into one "this-hop pick
        # accuracy" number; that pooled number alone can't tell whether a drop at a given hop
        # comes from the model genuinely struggling, or just from the active pool's source mix
        # shifting toward a noisier-labeled source at that hop (see module docstring on mixed mode).
        n_correct_by_src: dict[tuple[str, str], int] = defaultdict(int)
        n_total_by_src: dict[tuple[str, str], int] = defaultdict(int)
        answer_prompts: list[str] = []
        answer_meta: list[tuple[ExampleState, str, str, dict]] = []
        with torch.no_grad():
            for i, (ex, raw_sq, expanded_q, paras) in enumerate(zip(active, raw_sqs, expanded_qs, candidates_per_ex)):
                if not paras:
                    continue
                feats = group_feats_by_ex[i]
                scores = model(feats)
                pick = int(torch.argmax(scores).item())
                # gold_local_by_i[i] is already the right comparison for either mode: for gt
                # examples it's the positionally-resolved pool slot (equivalent to comparing
                # para idx against gold_idxs[hop_j-1]); for bart_decompose examples it's the
                # LLM-judge-derived target, so "correct" there means the model's real pick
                # agrees with that judge-derived label, not a dataset-annotated position (see
                # module docstring).
                src_key = (ex.dataset, ex.decompose_mode)
                n_total_by_src[src_key] += 1
                if gold_local_by_i.get(i) == pick:
                    n_correct += 1
                    n_correct_by_src[src_key] += 1
                answer_prompts.append(
                    prompt_short_answer_with_context(ex.q_main, ex.beam.hop_steps, ex.beam.prior, expanded_q, para)
                )
                answer_meta.append((ex, expanded_q, (para.get("paragraph_text") or "").strip(), para))
        hop_acc = n_correct / len(active) if active else 0.0
        acc_by_src = {
            f"{ds}/{mode}": round(n_correct_by_src[(ds, mode)] / n, 4)
            for (ds, mode), n in n_total_by_src.items() if n
        }
        print(f"[hop {hop_j}] this-hop pick accuracy (model's own picks vs gold): {hop_acc:.4f}"
              f"  by source: {acc_by_src}", flush=True)

        # Real short-answer generation, same as production -- hop_{j+1}'s query depends on
        # what the model actually picked here, including if it's wrong.
        answer_raw = (
            generator.generate_batch(answer_prompts, args.max_new_tokens_answer, desc=f"answer-hop{hop_j}")
            if answer_prompts else []
        )
        for (ex, expanded_q, ev_text, para), raw in zip(answer_meta, answer_raw):
            sub_ans = normalize_short_answer(raw)
            ex.beam.hop_steps.append((expanded_q, ev_text))
            ex.beam.prior.append(sub_ans)
            ex.beam.para_ids.append(int(para.get("idx", -1)))

        elapsed = time.perf_counter() - t0
        per_hop_stats.append({
            "hop": hop_j, "n_active": len(active),
            "n_train_groups_added": len(new_train_groups), "n_val_groups_added": len(new_val_groups),
            "n_train_groups_total": len(accumulated_train_groups), "n_val_groups_total": n_val_total,
            "epochs_run": epochs_run,
            "best_val_accuracy": round(best_val_metric, 4) if best_state is not None else None,
            "pick_accuracy": round(hop_acc, 4), "pick_accuracy_by_source": acc_by_src,
            "elapsed_min": round(elapsed / 60, 2),
        })
        print(f"[hop {hop_j}] elapsed {elapsed / 60:.1f} min", flush=True)

    save_artifact(args.out_artifact, feature_pca, model, meta={
        "pca_dim": args.pca_dim, "layer_by_j": DEFAULT_PER_J_LAYERS,
        "trained_on": {"decompose_mode": args.decompose_mode, "split": args.split,
                        "n_examples": len(examples), "sources": source_summary},
        "per_hop_stats": per_hop_stats,
    })
    run_wall_sec = time.perf_counter() - run_t0
    (out_dir / "run_meta.json").write_text(json.dumps({
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "per_hop_stats": per_hop_stats,
        "wall_clock_hr": round(run_wall_sec / 3600, 3),
        "artifact": str(args.out_artifact),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved frozen artifact -> {args.out_artifact}")
    print(f"Run meta -> {out_dir / 'run_meta.json'}")
    print(f"Wall clock hr {run_wall_sec / 3600:.3f}")


if __name__ == "__main__":
    main()
