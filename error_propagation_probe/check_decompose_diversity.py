#!/usr/bin/env python3
"""
Prerequisite check for update_doc/0713/0713update.md section 4 ("fold
decompose into the beam"): before building anything, verify that BART's
beam-search candidates beyond top-1 are actually (a) diverse from each other
and (b) sometimes closer to the gold decomposition than top-1 is. If the
top-N candidates are near-duplicates of top-1, or a better one basically
never shows up beyond top-1, there is nothing for a decompose-level beam
search to exploit and the whole idea is moot.

Reuses (imports, does not copy) decompose/bart/data_utils.py's
record loading / hop parsing and predict.py's model-loading pattern.
Read-only against the existing checkpoint — no training, no changes to
decompose/bart/.

Usage:
  python check_decompose_diversity.py --limit 30 --num-beams 10
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
BART_DIR = PROJECT_ROOT / "decompose" / "bart"
sys.path.insert(0, str(BART_DIR))

from data_utils import (  # noqa: E402
    decomposed_text_from_record,
    default_predict_file_for_dataset,
    iter_prediction_records,
    record_question,
    setup_bart_tokenizer,
    target_to_hops,
)


def sentence_bleu(hyp: str, ref: str) -> float:
    import sacrebleu
    if not hyp.strip() or not ref.strip():
        return 0.0
    return float(sacrebleu.sentence_bleu(hyp, [ref]).score)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-dir", type=Path,
        default=BART_DIR / "outputs" / "bart_decomposer_musique_2wiki_repro" / "checkpoint-1425",
    )
    ap.add_argument(
        "--dev-file", type=Path,
        default=BART_DIR / "data" / "musique_raw" / "musique_ans_gold_context_version_dev.jsonl",
        help="Must have composed_question_text/component_question_texts fields "
             "(default_predict_file_for_dataset('musique') points at the raw benchmark "
             "file instead, which lacks these and silently yields 0 records).",
    )
    ap.add_argument("--dataset", choices=("musique",), default="musique")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--num-beams", type=int, default=10)
    ap.add_argument("--max-source-length", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-checkpoints", type=int, nargs="+", default=[1, 3, 5, 10],
                     help="Report best-of-N at these N values (must be <= --num-beams)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.dev_file is None:
        args.dev_file = default_predict_file_for_dataset(args.dataset)

    import torch
    from transformers import BartForConditionalGeneration, BartTokenizer

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if (args.model_dir / "tokenizer_config.json").exists():
        tokenizer = BartTokenizer.from_pretrained(args.model_dir)
    else:
        tokenizer = setup_bart_tokenizer(str(args.model_dir))
    model = BartForConditionalGeneration.from_pretrained(args.model_dir)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    records = list(iter_prediction_records(args.dev_file, dataset=args.dataset, limit=args.limit))
    print(f"Loaded {len(records)} records from {args.dev_file}")
    print(f"Model: {args.model_dir}  num_beams={args.num_beams}  device={device}")

    n_checkpoints = sorted(set(n for n in args.n_checkpoints if n <= args.num_beams))

    # Per-example results.
    best_bleu_at_n: dict[int, list[float]] = {n: [] for n in n_checkpoints}
    exact_hit_at_n: dict[int, list[bool]] = {n: [] for n in n_checkpoints}
    n_distinct_candidates: list[int] = []
    avg_pairwise_bleu: list[float] = []
    top1_bleu: list[float] = []

    for i, record in enumerate(records):
        source = record_question(record, args.dataset)
        gold_target = decomposed_text_from_record(record)
        encoded = tokenizer(
            [source], add_special_tokens=False, max_length=args.max_source_length,
            truncation=True, padding=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams, num_return_sequences=args.num_beams,
            )
        candidates = tokenizer.batch_decode(generated, skip_special_tokens=True)

        bleus = [sentence_bleu(c, gold_target) for c in candidates]
        exacts = [c.strip() == gold_target.strip() for c in candidates]
        top1_bleu.append(bleus[0])

        for n in n_checkpoints:
            best_bleu_at_n[n].append(max(bleus[:n]))
            exact_hit_at_n[n].append(any(exacts[:n]))

        distinct = len(set(c.strip() for c in candidates))
        n_distinct_candidates.append(distinct)

        pairwise = []
        for a in range(len(candidates)):
            for b in range(a + 1, len(candidates)):
                pairwise.append(sentence_bleu(candidates[a], candidates[b]))
        if pairwise:
            avg_pairwise_bleu.append(statistics.mean(pairwise))

        if i < 5:
            print(f"\n--- example {i} (id={record.get('id')}) ---")
            print(f"  gold:  {target_to_hops(gold_target)}")
            for rank, (c, b) in enumerate(zip(candidates[:5], bleus[:5])):
                print(f"  #{rank} (bleu={b:.1f}): {target_to_hops(c)}")

    print("\n" + "=" * 60)
    print(f"n_examples = {len(records)}")
    print(f"avg top-1 BLEU              = {statistics.mean(top1_bleu):.2f}")
    print(f"avg #distinct candidates / {args.num_beams} beams = {statistics.mean(n_distinct_candidates):.2f}")
    print(f"avg pairwise BLEU among candidates (lower = more diverse) = "
          f"{statistics.mean(avg_pairwise_bleu):.2f}" if avg_pairwise_bleu else "n/a")
    print("\nBest-of-N (oracle, does a closer-to-gold candidate exist beyond top-1?):")
    print(f"{'N':>4} {'best_bleu_mean':>15} {'exact_hit_rate':>15}")
    for n in n_checkpoints:
        print(f"{n:>4} {statistics.mean(best_bleu_at_n[n]):>15.2f} "
              f"{sum(exact_hit_at_n[n]) / len(exact_hit_at_n[n]):>15.4f}")


if __name__ == "__main__":
    main()
