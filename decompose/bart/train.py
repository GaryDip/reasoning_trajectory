#!/usr/bin/env python3
"""Train official-format MuSiQue BART decomposer with HuggingFace + modern PyTorch."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from data_utils import (
    DEFAULT_MODEL_NAME,
    DEFAULT_RAW_DEV_FILE,
    DEFAULT_RAW_TRAIN_FILE,
    decomposed_text_from_record,
    iter_raw_records,
    record_to_pair,
    setup_bart_tokenizer,
    target_to_hops,
    wrap_target_with_special_tokens,
)
from metrics import (
    decode_generated_sequences,
    flatten_metrics_for_trainer,
    format_eval_summary,
    summarize_by_gold_hops,
)


@dataclass
class PairExample:
    id: str
    source: str
    target: str


class DecomposeDataset(Dataset):
    def __init__(
        self,
        examples: list[PairExample],
        tokenizer,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        model_inputs = self.tokenizer(
            example.source,
            add_special_tokens=False,
            max_length=self.max_source_length,
            truncation=True,
        )
        wrapped_target = wrap_target_with_special_tokens(self.tokenizer, example.target)
        labels = self.tokenizer(
            wrapped_target,
            add_special_tokens=False,
            max_length=self.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


class StratifiedEvalCallback(TrainerCallback):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.latest_summary: dict | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.latest_summary:
            return

        payload = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "metrics": metrics or {},
            **self.latest_summary,
        }
        out_path = self.output_dir / f"eval_by_hop_step_{state.global_step}.json"
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

        latest_path = self.output_dir / "eval_by_hop_latest.json"
        with open(latest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

        history_path = self.output_dir / "eval_by_hop_history.jsonl"
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        print(
            f"\n[eval step={state.global_step} epoch={state.epoch:.2f}]\n"
            f"{format_eval_summary(self.latest_summary, prefix='  ')}\n",
            flush=True,
        )


def build_examples(paths: list[Path], limit: int) -> list[PairExample]:
    examples: list[PairExample] = []
    for path in paths:
        for record in iter_raw_records(path, limit=limit):
            source, target = record_to_pair(record)
            examples.append(PairExample(id=str(record.get("id")), source=source, target=target))
    return examples


def make_compute_metrics(tokenizer, eval_examples: list[PairExample], eval_callback: StratifiedEvalCallback):
    gold_hop_counts = [len(target_to_hops(example.target)) for example in eval_examples]
    gold_targets = [example.target for example in eval_examples]

    def compute_metrics(eval_pred):
        preds, _labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = decode_generated_sequences(tokenizer, preds)
        n = min(len(decoded_preds), len(gold_targets))
        summary = summarize_by_gold_hops(
            decoded_preds[:n],
            gold_targets[:n],
            gold_hop_counts[:n],
        )
        eval_callback.latest_summary = summary
        return flatten_metrics_for_trainer(summary)

    return compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, nargs="+", default=[DEFAULT_RAW_TRAIN_FILE])
    parser.add_argument("--dev-file", type=Path, nargs="+", default=[DEFAULT_RAW_DEV_FILE])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bart_decomposer"))
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-source-length", type=int, default=100)
    parser.add_argument("--max-target-length", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval-beams", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Debug: cap train/dev examples")
    parser.add_argument("--no-eval", action="store_true", help="Skip dev evaluation")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def serialize_paths(paths: list[Path]) -> str | list[str]:
    serialized = [str(path) for path in paths]
    return serialized[0] if len(serialized) == 1 else serialized


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = build_examples(args.train_file, args.limit)
    if not train_examples:
        raise SystemExit(f"No training examples found in {serialize_paths(args.train_file)}")

    eval_examples: list[PairExample] = []
    if not args.no_eval:
        eval_examples = build_examples(args.dev_file, args.limit)

    tokenizer = setup_bart_tokenizer(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = DecomposeDataset(
        train_examples,
        tokenizer,
        args.max_source_length,
        args.max_target_length,
    )
    eval_dataset = None
    if eval_examples:
        eval_dataset = DecomposeDataset(
            eval_examples,
            tokenizer,
            args.max_source_length,
            args.max_target_length,
        )

    eval_callback = StratifiedEvalCallback(args.output_dir)
    callbacks: list[TrainerCallback] = []
    if eval_dataset is not None:
        callbacks.append(eval_callback)
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    use_cuda = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        label_smoothing_factor=args.label_smoothing,
        lr_scheduler_type="polynomial",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch" if eval_dataset is not None else "steps",
        save_steps=500,
        logging_steps=1,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.eval_beams,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="bleu" if eval_dataset is not None else None,
        greater_is_better=True,
        seed=args.seed,
        fp16=args.fp16 and use_cuda and not args.bf16,
        bf16=args.bf16 and use_cuda,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=(
            make_compute_metrics(tokenizer, eval_examples, eval_callback)
            if eval_dataset is not None
            else None
        ),
        callbacks=callbacks or None,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    meta = {
        "format": "musique_official_raw",
        "data_fields": {
            "source": "composed_question_text",
            "target": "component_question_texts",
        },
        "target_format": "[[CQS]] i [[CQE]] hop [[RQS]] j [[RQE]] ... with [SEP]",
        "model_name": args.model_name,
        "train_file": serialize_paths(args.train_file),
        "dev_file": serialize_paths(args.dev_file),
        "num_train_examples": len(train_examples),
        "num_dev_examples": len(eval_examples),
        "extra_tokens": "CQS/CQE and RQS/RQE indices 0-4",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "label_smoothing": args.label_smoothing,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "max_grad_norm": args.max_grad_norm,
        "patience": args.patience,
        "eval_beams": args.eval_beams,
        "seed": args.seed,
        "limit": args.limit,
    }
    with open(args.output_dir / "train_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
