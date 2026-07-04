#!/usr/bin/env python3
"""Convert decomposer predict output from raw markers to MuSiQue v1.0 hop text format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from data_utils import raw_decomposition_to_v1_hop_texts, translate_id


def normalize_row_id(row_id: str) -> str:
    if row_id.startswith("double__") or row_id.startswith("triple_") or row_id.startswith("quadruple_"):
        return translate_id(row_id)
    return row_id


def build_v1_decomposition_record(
    prediction_row: dict,
    *,
    include_gold: bool = False,
) -> dict:
    row_id = str(prediction_row.get("id") or "")
    pred_hops_v1 = raw_decomposition_to_v1_hop_texts(prediction_row["predicted_target"])
    record = {
        "id": normalize_row_id(row_id),
        "raw_id": row_id,
        "question": prediction_row.get("question") or "",
        "predicted_question_decomposition": [{"question": hop} for hop in pred_hops_v1],
        "num_hops_pred": len(pred_hops_v1),
    }
    for key in ("dataset", "answer", "type", "level", "num_hops_gold"):
        if key in prediction_row:
            record[key] = prediction_row[key]
    if include_gold and prediction_row.get("gold_target"):
        gold_hops_v1 = raw_decomposition_to_v1_hop_texts(prediction_row["gold_target"])
        record["gold_question_decomposition"] = [{"question": hop} for hop in gold_hops_v1]
        record["num_hops_gold"] = len(gold_hops_v1)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/musique_ans_dev_predictions.jsonl"),
        help="predict.py output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/musique_ans_dev_predictions_v1.jsonl"),
    )
    parser.add_argument(
        "--include-gold",
        action="store_true",
        help="Also emit gold_question_decomposition in v1 hop text format",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(args.input, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    converted: list[dict] = []
    errors: list[tuple[str, str]] = []
    for row in tqdm(rows, desc="convert", unit="row"):
        try:
            converted.append(
                build_v1_decomposition_record(row, include_gold=args.include_gold)
            )
        except Exception as exc:
            errors.append((str(row.get("id")), str(exc)))

    with open(args.output, "w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(converted)} rows to {args.output}")
    if errors:
        print(f"Skipped {len(errors)} rows due to conversion errors:")
        for row_id, message in errors[:5]:
            print(f"  {row_id}: {message}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


if __name__ == "__main__":
    main()
