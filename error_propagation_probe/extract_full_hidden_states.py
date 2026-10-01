#!/usr/bin/env python3
"""
Same forward-pass machinery as hidden_states/extract_hidden_states.py, reused
by import — the only difference is that every trace is taken to its FULL
K+1 prefixes (h_0..h_K), never truncated at n_take = wrong_hop + 1. That
truncation (production behavior, kept as-is in extract_hidden_states.py) is
exactly why no data exists anywhere for "hidden state after a hop that comes
AFTER the injected error" — this script exists to capture that, for the
multi-error trace set built by build_multi_error_traces.py.

Output layout mirrors hidden_states/{split}/ (activations/{pos,neg}/*.npz +
manifest.jsonl), just under this folder, so it never touches production
hidden_states/ data.

Usage:
  python extract_full_hidden_states.py --split dev \
      --trace-file data/musique/dev_multi_error.jsonl --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
HIDDEN_STATES_DIR = PROJECT_ROOT / "hidden_states"
sys.path.insert(0, str(HIDDEN_STATES_DIR))

from extract_hidden_states import (  # noqa: E402
    HiddenExtractor,
    PrefixWork,
    TraceJob,
    prefix_token_len,
    save_job_npz,
)
from trace_parse import cumulative_prefix_strings, parse_trace_structure  # noqa: E402


def full_trace_job_from_row(row: dict[str, Any]) -> TraceJob:
    """Like TraceJob.from_row, but n_take is ALWAYS all prefixes — the one
    line that differs from production (which sets n_take = wrong_hop + 1 for
    error traces)."""
    trace = (row.get("reasoning_trace") or "").strip()
    if not trace:
        raise ValueError("empty reasoning_trace")
    q, blocks, _ans = parse_trace_structure(trace)
    prefixes = cumulative_prefix_strings(q, blocks)
    trace_type = str(row.get("trace_type", "correct"))
    label = "positive" if trace_type != "error" else "negative"
    subdir = "pos" if trace_type != "error" else "neg"
    row_k = row.get("K")
    k_out = int(row_k) if row_k is not None else len(blocks)
    return TraceJob(
        row=row, rid=str(row.get("id", "")).strip(), label=label, subdir=subdir,
        k=k_out, n_take=len(prefixes), prefixes=prefixes,
    )


def load_trace_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_done_ids(manifest_path: Path) -> set[str]:
    done: set[str] = set()
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(str(json.loads(line)["id"]))
    return done


def main() -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"Install torch + transformers: {exc}") from exc

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("train", "dev"), default="dev")
    ap.add_argument("--trace-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--trace-chunk-size", type=int, default=64)
    ap.add_argument("--length-bucket-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not args.trace_file.is_file():
        sys.exit(f"Trace file not found: {args.trace_file}")

    split_dir = args.out_dir or (HERE / "hidden_states_full" / args.split)
    activ_root = split_dir / "activations"
    manifest_path = split_dir / "manifest.jsonl"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    rows = load_trace_rows(args.trace_file, args.limit)
    done_ids = load_done_ids(manifest_path) if args.resume else set()
    todo_rows = [r for r in rows if str(r.get("id", "")).strip() not in done_ids]

    print("=" * 60)
    print("full-prefix hidden state extraction (error_propagation_probe)")
    print(f"  trace:  {args.trace_file}")
    print(f"  out:    {split_dir}")
    print(f"  model:  {args.model} layer={args.layer}")
    print(f"  device: {device} dtype={args.dtype}")
    print(f"  rows:   {len(todo_rows)} / {len(rows)} (skip {len(done_ids)} done)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dev_map = {"": device} if device != "cpu" else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype], device_map=dev_map,
    )
    model.eval()
    if device == "cpu":
        model.to(torch.device("cpu"))

    extractor = HiddenExtractor(model, tokenizer, layer=args.layer)

    activ_root.mkdir(parents=True, exist_ok=True)
    (activ_root / "pos").mkdir(exist_ok=True)
    (activ_root / "neg").mkdir(exist_ok=True)

    manifest_mode = "a" if args.resume and manifest_path.exists() else "w"
    n_ok = n_err = 0

    with open(manifest_path, manifest_mode, encoding="utf-8") as mf:
        pbar = tqdm(total=len(todo_rows), desc="extract", unit="ex")
        for chunk_start in range(0, len(todo_rows), args.trace_chunk_size):
            chunk_rows = todo_rows[chunk_start : chunk_start + args.trace_chunk_size]
            jobs: list[TraceJob] = []
            works: list[PrefixWork] = []
            row_by_job: list[tuple[dict[str, Any], TraceJob]] = []

            for row in chunk_rows:
                rid = str(row.get("id", "")).strip()
                try:
                    job = full_trace_job_from_row(row)
                    jobs.append(job)
                    row_by_job.append((row, job))
                    for i, pfx in enumerate(job.prefixes[: job.n_take]):
                        works.append(PrefixWork(job, i, pfx, token_len=prefix_token_len(tokenizer, pfx)))
                except Exception as exc:
                    n_err += 1
                    pbar.update(1)
                    tqdm.write(json.dumps({"error": str(exc), "id": rid}, ensure_ascii=False))

            if works:
                try:
                    extractor.run_work(works, batch_size=args.batch_size,
                                        length_bucket_tokens=args.length_bucket_tokens)
                except Exception as exc:
                    for row, job in row_by_job:
                        n_err += 1
                        pbar.update(1)
                        tqdm.write(json.dumps({"error": str(exc), "id": job.rid}, ensure_ascii=False))
                    continue

            for row, job in row_by_job:
                try:
                    extractor.finalize_job(job)
                    out_path = save_job_npz(job, split_dir)
                    mf.write(json.dumps({
                        "id": job.rid,
                        "label": job.label,
                        "trace_type": row.get("trace_type"),
                        "K": job.k,
                        "wrong_hops": row.get("wrong_hops") or [],
                        "n_wrong": int(row.get("n_wrong", len(row.get("wrong_hops") or []))),
                        "num_hops": job.k,
                        "num_prefixes_saved": len(job.hiddens),
                        "model_layer_0based": args.layer,
                        "path": str(out_path.relative_to(split_dir)),
                        "source_id": row.get("source_id"),
                    }, ensure_ascii=False) + "\n")
                    n_ok += 1
                except Exception as exc:
                    n_err += 1
                    tqdm.write(json.dumps({"error": str(exc), "id": job.rid}, ensure_ascii=False))
                pbar.update(1)
        pbar.close()

    meta = {
        "model": args.model, "layer_0based": args.layer, "dtype": args.dtype,
        "full_forward_all_prefixes": True, "n_ok": n_ok, "n_err": n_err,
        "trace_file": str(args.trace_file), "manifest": str(manifest_path),
    }
    (split_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"done ok={n_ok} err={n_err} -> {split_dir}")


if __name__ == "__main__":
    main()
