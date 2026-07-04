#!/usr/bin/env python3
"""Run BART decomposer inference on MuSiQue, 2WikiQA, or HotpotQA questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from data_utils import (
    build_inference_prediction_record,
    default_predict_file_for_dataset,
    infer_dataset_name,
    iter_prediction_records,
    record_question,
    setup_bart_tokenizer,
)
from metrics import format_eval_summary, summarize_prediction_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/bart_decomposer"),
        help="Fine-tuned model directory (Trainer output or checkpoint)",
    )
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=None,
        help="Input file. Defaults to the dev file for --dataset.",
    )
    parser.add_argument(
        "--dataset",
        choices=("auto", "musique", "2wiki", "hotpot"),
        default="auto",
        help="Input schema. auto infers from --dev-file path; omitted --dev-file defaults to MuSiQue.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dev_predictions.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--max-source-length", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dev_file is None:
        dataset = "musique" if args.dataset == "auto" else args.dataset
        args.dev_file = default_predict_file_for_dataset(dataset)
    args.dataset = infer_dataset_name(args.dev_file, args.dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer: BartTokenizer
    if (args.model_dir / "tokenizer_config.json").exists():
        tokenizer = BartTokenizer.from_pretrained(args.model_dir)
    else:
        tokenizer = setup_bart_tokenizer(str(args.model_dir))

    model = BartForConditionalGeneration.from_pretrained(args.model_dir)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    records = list(
        iter_prediction_records(args.dev_file, dataset=args.dataset, limit=args.limit)
    )
    if not records:
        raise SystemExit(f"No records found in {args.dev_file}")

    outputs: list[dict] = []
    batch_starts = range(0, len(records), args.batch_size)
    for start in tqdm(
        batch_starts,
        desc="predict",
        unit="batch",
        total=(len(records) + args.batch_size - 1) // args.batch_size,
    ):
        batch = records[start : start + args.batch_size]
        sources = [record_question(record, args.dataset) for record in batch]
        encoded = tokenizer(
            sources,
            add_special_tokens=False,
            max_length=args.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
        }
        if args.fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda"):
                generated = model.generate(**encoded, **generate_kwargs)
        else:
            generated = model.generate(**encoded, **generate_kwargs)

        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for record, text in zip(batch, texts):
            outputs.append(
                build_inference_prediction_record(record, text, dataset=args.dataset)
            )

    with open(args.output, "w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = args.output.with_suffix(".summary.json")
    meta_path = args.output.with_suffix(".meta.json")
    has_gold = all("gold_target" in row for row in outputs)
    if has_gold:
        summary = summarize_prediction_rows(outputs)
    else:
        hop_dist: dict[int, int] = {}
        gold_hop_dist: dict[int, int] = {}
        hop_count_correct = 0
        hop_count_total = 0
        for row in outputs:
            hop_dist[row["num_hops_pred"]] = hop_dist.get(row["num_hops_pred"], 0) + 1
            if row.get("num_hops_gold") is not None:
                gold_k = int(row["num_hops_gold"])
                gold_hop_dist[gold_k] = gold_hop_dist.get(gold_k, 0) + 1
                hop_count_total += 1
                if row["num_hops_pred"] == gold_k:
                    hop_count_correct += 1
        summary = {
            "n": len(outputs),
            "has_gold": False,
            "pred_hop_distribution": dict(sorted(hop_dist.items())),
        }
        if hop_count_total:
            summary.update(
                {
                    "gold_hop_distribution": dict(sorted(gold_hop_dist.items())),
                    "hop_count_acc": hop_count_correct / hop_count_total,
                    "hop_count_total": hop_count_total,
                    "gold_hop_definition": (
                        "2wiki: len(evidences) if present, otherwise unique supporting_facts titles; "
                        "hotpot: unique supporting_facts titles"
                    ),
                }
            )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    meta = {
        "format": f"{args.dataset}_question_decomposition",
        "dataset": args.dataset,
        "model_dir": str(args.model_dir),
        "dev_file": str(args.dev_file),
        "output": str(args.output),
        "summary": str(summary_path),
        "num_examples": len(outputs),
        "batch_size": args.batch_size,
        "num_beams": args.num_beams,
        "max_source_length": args.max_source_length,
        "max_new_tokens": args.max_new_tokens,
        "device": str(device),
    }
    if has_gold:
        meta.update(
            {
                "overall_bleu": summary["bleu"],
                "overall_exact_match": summary["exact_match"],
                "overall_hop_count_acc": summary["hop_count_acc"],
            }
        )
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(outputs)} predictions to {args.output}")
    if has_gold:
        print(format_eval_summary(summary, prefix="  "))
    else:
        msg = f"  prediction-only: n={summary['n']} | pred_hops={summary['pred_hop_distribution']}"
        if "hop_count_acc" in summary:
            msg += (
                f" | gold_hops={summary['gold_hop_distribution']}"
                f" | hop_count_acc={summary['hop_count_acc']:.1%}"
            )
        print(msg)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
