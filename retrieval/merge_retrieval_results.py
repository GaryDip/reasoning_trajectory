#!/usr/bin/env python3
"""
Merge new method results into an existing retrieval experiment output.

Use case: you already have retrieval_exp_dev_full.json from baseline/lr_rerank/...
and later ran only --methods colbert. This script adds the colbert block into
the old metrics JSON and patches each row in the cases JSONL.

Usage
-----
  # 1) run colbert-only (same data split / filters as the base run)
  python run_retrieval_exp.py --methods colbert --run-tag colbert --device cuda:2

  # 2) merge into the full result files (creates .bak backups first)
  python merge_retrieval_results.py \
      --base-json  results/retrieval_exp_dev_full.json \
      --add-json   results/retrieval_exp_dev_colbert.json \
      --base-cases results/retrieval_cases_dev_full.jsonl \
      --add-cases  results/retrieval_cases_dev_colbert.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

CASE_DICT_FIELDS = (
    "chain_recall1",
    "chain_recall3",
    "chain_selection",
    "predicted_answers",
    "answer_em",
    "answer_f1",
    "gate_fired_any",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _method_keys(data: dict) -> list[str]:
    skip = {"config", "n_skip"}
    return sorted(k for k in data if k not in skip and isinstance(data[k], dict))


def merge_metrics(base: dict, add: dict, methods: list[str]) -> dict:
    out = dict(base)
    for m in methods:
        if m not in add:
            raise KeyError(f"method {m!r} missing in add-json")
        if m in out and m not in {"config", "n_skip"}:
            print(f"  [warn] overwriting existing method block: {m}", file=sys.stderr)
        out[m] = add[m]

    base_cfg = dict(out.get("config") or {})
    add_cfg = dict(add.get("config") or {})
    merged_methods = list(base_cfg.get("methods") or _method_keys(base))
    for m in methods:
        if m not in merged_methods:
            merged_methods.append(m)
    base_cfg["methods"] = merged_methods
    for key, val in add_cfg.items():
        if key == "methods":
            continue
        if key not in base_cfg:
            base_cfg[key] = val
    out["config"] = base_cfg
    if base.get("n_skip") != add.get("n_skip"):
        print(
            f"  [warn] n_skip differs: base={base.get('n_skip')} add={add.get('n_skip')}",
            file=sys.stderr,
        )
    return out


def merge_cases(base_rows: list[dict], add_rows: list[dict], methods: list[str]) -> list[dict]:
    add_by_id = {row["id"]: row for row in add_rows}
    missing: list[str] = []
    out_rows: list[dict] = []

    for base in base_rows:
        eid = base["id"]
        add = add_by_id.get(eid)
        if add is None:
            missing.append(eid)
            out_rows.append(base)
            continue

        merged = dict(base)
        for field in CASE_DICT_FIELDS:
            if field not in add:
                continue
            dst = dict(merged.get(field) or {})
            for m in methods:
                if m in add[field]:
                    dst[m] = add[field][m]
            merged[field] = dst

        base_hops = merged.get("hop_results") or []
        add_hops = add.get("hop_results") or []
        if len(base_hops) != len(add_hops):
            print(
                f"  [warn] hop count mismatch id={eid}: "
                f"base={len(base_hops)} add={len(add_hops)}",
                file=sys.stderr,
            )
        for i, hop in enumerate(base_hops):
            if i >= len(add_hops):
                break
            for m in methods:
                if m in add_hops[i]:
                    hop[m] = add_hops[i][m]

        out_rows.append(merged)

    extra_ids = sorted(set(add_by_id) - {r["id"] for r in base_rows})
    if missing:
        print(f"  [warn] {len(missing)} base ids missing in add-cases", file=sys.stderr)
    if extra_ids:
        print(f"  [warn] {len(extra_ids)} add-only ids ignored", file=sys.stderr)
    return out_rows


def _backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    print(f"  backup -> {bak}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge retrieval_exp method results into base files.")
    ap.add_argument("--base-json", type=Path, required=True)
    ap.add_argument("--add-json", type=Path, required=True)
    ap.add_argument("--base-cases", type=Path, required=True)
    ap.add_argument("--add-cases", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Default: overwrite --base-json")
    ap.add_argument("--out-cases", type=Path, default=None,
                    help="Default: overwrite --base-cases")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="Methods to merge from add files (default: all method keys in add-json)")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    base = _load_json(args.base_json)
    add = _load_json(args.add_json)
    methods = args.methods or _method_keys(add)
    if not methods:
        sys.exit("No methods found in add-json.")

    print(f"Merging methods: {methods}", file=sys.stderr)
    merged = merge_metrics(base, add, methods)

    base_rows = _load_jsonl(args.base_cases)
    add_rows = _load_jsonl(args.add_cases)
    merged_rows = merge_cases(base_rows, add_rows, methods)

    out_json = args.out_json or args.base_json
    out_cases = args.out_cases or args.base_cases
    if not args.no_backup:
        if out_json.resolve() == args.base_json.resolve():
            _backup(args.base_json)
        if out_cases.resolve() == args.base_cases.resolve():
            _backup(args.base_cases)

    out_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    with out_cases.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote metrics -> {out_json}", file=sys.stderr)
    print(f"Wrote cases   -> {out_cases} ({len(merged_rows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
