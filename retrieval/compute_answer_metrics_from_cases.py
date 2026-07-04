#!/usr/bin/env python3
"""Compute final-answer EM/F1 from retrieval case JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_INFER = ROOT.parent / "llama_infer_reasoning"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE_INFER))

from run_musique_pipeline import judge_answer_official  # noqa: E402
from run_retrieval_exp import (  # noqa: E402
    HOTPOT_DEV_FILE,
    MUSIQUE_DIR,
    TWOWIKI_DEV_FILE,
    load_2wiki,
    load_hotpot,
    load_musique,
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_records(dataset: str, split: str) -> dict[str, dict[str, Any]]:
    if dataset == "musique":
        return load_musique(MUSIQUE_DIR, split)
    if dataset == "2wiki":
        if split != "dev":
            raise SystemExit("2wiki loader currently expects --split dev")
        return load_2wiki(TWOWIKI_DEV_FILE)
    if dataset == "hotpot":
        if split != "dev":
            raise SystemExit("hotpot loader currently expects --split dev")
        return load_hotpot(HOTPOT_DEV_FILE)
    raise SystemExit(f"Unknown dataset: {dataset}")


def infer_methods(rows: list[dict[str, Any]]) -> list[str]:
    methods: set[str] = set()
    for row in rows:
        methods.update((row.get("predicted_answers") or {}).keys())
    return sorted(methods)


def compute_metrics(
    rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    methods: list[str],
) -> dict[str, dict[str, Any]]:
    stats = {
        method: {"n_examples": 0, "answer_em": 0, "answer_f1": 0.0, "missing_gold": 0}
        for method in methods
    }

    for row in rows:
        record = records.get(str(row.get("id")))
        if record is None:
            for method in methods:
                stats[method]["missing_gold"] += 1
            continue
        predicted_answers = row.get("predicted_answers") or {}
        for method in methods:
            if method not in predicted_answers:
                continue
            predicted = str(predicted_answers.get(method) or "")
            em, f1 = judge_answer_official(predicted, record)
            stats[method]["n_examples"] += 1
            stats[method]["answer_em"] += int(em)
            stats[method]["answer_f1"] += float(f1)

    out: dict[str, dict[str, Any]] = {}
    for method, st in stats.items():
        n = st["n_examples"]
        out[method] = {
            "n_examples": n,
            "answer_em": round(st["answer_em"] / n, 4) if n else 0.0,
            "answer_f1": round(st["answer_f1"] / n, 4) if n else 0.0,
        }
        if st["missing_gold"]:
            out[method]["missing_gold"] = st["missing_gold"]
    return out


def patch_metrics_json(path: Path, answer_metrics: dict[str, dict[str, Any]]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    for method, metrics in answer_metrics.items():
        if method not in data or not isinstance(data[method], dict):
            continue
        data[method]["answer_overall"] = metrics
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("musique", "2wiki", "hotpot"), required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--patch-json", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_cases(args.cases)
    if not rows:
        raise SystemExit(f"No cases found in {args.cases}")
    methods = args.methods or infer_methods(rows)
    records = load_records(args.dataset, args.split)
    answer_metrics = compute_metrics(rows, records, methods)

    result = {
        "dataset": args.dataset,
        "split": args.split,
        "cases": str(args.cases),
        "n_cases": len(rows),
        "answer_metrics": answer_metrics,
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.patch_json:
        if args.metrics_json is None:
            raise SystemExit("--patch-json requires --metrics-json")
        patch_metrics_json(args.metrics_json, answer_metrics)
        print(f"Patched answer_overall in {args.metrics_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
