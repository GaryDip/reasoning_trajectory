"""Load raw records and decompose index for MuSiQue / 2Wiki / Hotpot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DECOMPOSE_ROOT = PROJECT_ROOT / "data" / "decompose"


def get_decompose_enhance_path(dataset: str, split: str) -> Path:
    """LLM-paraphrased augmentations (train only for now)."""
    if dataset != "musique":
        raise ValueError(f"Enhance decompose only for musique, got {dataset}")
    return DECOMPOSE_ROOT / "musique" / "gt" / f"{split}_nl_enhance.jsonl"


def load_decompose_bundle(
    base_path: Path,
    *,
    enhance_path: Path | None = None,
    include_enhance: bool = True,
    require_verify_pass: bool = True,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Load base (+ optional enhance) decompose index.

    Returns:
      sub_questions_by_id: decompose_id → sub_questions
      raw_id_by_decompose_id: decompose_id → MuSiQue raw record id
    """
    sub_qs: dict[str, list[str]] = {}
    raw_map: dict[str, str] = {}

    def _ingest(path: Path, *, enhanced: bool) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if enhanced and require_verify_pass and obj.get("verify_pass") is False:
                    continue
                did = str(obj.get("id", "")).strip()
                sqs = obj.get("sub_questions") or []
                if not did or not isinstance(sqs, list):
                    continue
                cleaned = [str(s) for s in sqs if str(s).strip()]
                if not cleaned:
                    continue
                sub_qs[did] = cleaned
                raw_map[did] = str(obj.get("source_id") or did).strip()

    _ingest(base_path, enhanced=False)

    if include_enhance and enhance_path is not None and enhance_path.is_file():
        _ingest(enhance_path, enhanced=True)

    return sub_qs, raw_map


def load_raw_index(dataset: str, split: str) -> dict[str, dict[str, Any]]:
    """Load all raw records keyed by id (for decompose job iteration)."""
    return {
        str(r.get("id", "")).strip(): r
        for r in iter_records(dataset, split)
        if str(r.get("id", "")).strip()
    }


def load_decompose_index(path: Path) -> dict[str, list[str]]:
    sub_qs, _ = load_decompose_bundle(path, include_enhance=False)
    return sub_qs


def iter_musique(raw_dir: Path, split: str) -> Iterator[dict[str, Any]]:
    path = raw_dir / f"musique_ans_v1.0_{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_2wiki(raw_dir: Path, split: str) -> Iterator[dict[str, Any]]:
    path = raw_dir / f"{split}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        yield from data
    else:
        yield from data.get("data", [])


def iter_hotpot(raw_dir: Path, split: str) -> Iterator[dict[str, Any]]:
    if split == "dev":
        path = raw_dir / "hotpot_dev_distractor_v1.json"
    else:
        path = raw_dir / "hotpot_train_v1.1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    yield from data


def get_raw_dir(dataset: str) -> Path:
    mapping = {
        "musique": RAW_ROOT / "musique",
        "2wiki": RAW_ROOT / "2wikimultihopqa",
        "hotpot": RAW_ROOT / "hotpotqa",
    }
    if dataset not in mapping:
        raise ValueError(f"Unknown dataset: {dataset}")
    return mapping[dataset]


def get_decompose_path(
    dataset: str,
    split: str,
    *,
    mode: str = "gt",
) -> Path:
    """
    Resolve decompose jsonl path.

    mode:
      gt   — MuSiQue GT NL annotations (train/dev); only musique has gt/
      bart — BART decomposer NL predictions (dev for all three datasets)
    """
    if mode == "gt":
        if dataset != "musique":
            raise ValueError(f"GT decompose only available for musique, got {dataset}")
        return DECOMPOSE_ROOT / "musique" / "gt" / f"{split}_nl.jsonl"
    if mode == "bart":
        if split != "dev":
            raise ValueError("BART predictions currently only copied for dev split")
        return DECOMPOSE_ROOT / dataset / "bart" / "dev_nl.jsonl"
    raise ValueError(f"Unknown decompose mode: {mode}")


def iter_records(dataset: str, split: str) -> Iterator[dict[str, Any]]:
    raw_dir = get_raw_dir(dataset)
    if dataset == "musique":
        yield from iter_musique(raw_dir, split)
    elif dataset == "2wiki":
        yield from iter_2wiki(raw_dir, split)
    elif dataset == "hotpot":
        yield from iter_hotpot(raw_dir, split)
    else:
        raise ValueError(dataset)
