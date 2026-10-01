#!/usr/bin/env python3
"""
Multi-layer version of extract_full_hidden_states.py, for testing whether gate v3's
anomaly score (which needs layers 15/23, not the single layer 31 the existing
hidden_states_full/ extraction has) rises with the NUMBER of wrong hops already in the
prefix (n_wrong), not just whether the current hop matches the one hop gate v3's training
labels were ever constructed around (see conversation).

Combines two pieces that each already exist, reused by import (neither is modified):
  - full_trace_job_from_row (error_propagation_probe/extract_full_hidden_states.py):
    every trace's FULL K+1 prefixes are extracted, never truncated at the first wrong hop
    -- required because build_multi_error_traces.py's rows can have wrong hops anywhere,
    including hops after the "first" one.
  - the multi-layer batched extraction machinery (MLJob/PrefixWork/run_extract_and_save/
    save_ml_job_npz/write_manifest_row/check_resume_layers_match, from
    hidden_states/extract_hidden_states_multilayer_pilot.py): one forward pass per prefix
    still reads out every requested layer at no extra cost.

Reads data/musique/{split}_multi_error.jsonl (already built by build_multi_error_traces.py,
already has an n_wrong field per row) -- no new trace construction needed, only new hidden
states. Output layout mirrors hidden_states_full/{split}/ under a separate
hidden_states_full_multilayer/{split}/ directory so it never touches the existing
layer-31-only extraction.

Usage:
  python extract_full_hidden_states_multilayer.py --split dev \
      --trace-file data/musique/dev_multi_error.jsonl --layers 7,15,23,31 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
HIDDEN_STATES_DIR = PROJECT_ROOT / "hidden_states"
sys.path.insert(0, str(HIDDEN_STATES_DIR))
sys.path.insert(0, str(HERE))

from extract_hidden_states import prefix_token_len, render_chat_text  # noqa: E402
from extract_hidden_states_multilayer_pilot import (  # noqa: E402
    MLJob,
    PrefixWork,
    check_resume_layers_match,
    run_extract_and_save,
)
from extract_full_hidden_states import (  # noqa: E402
    full_trace_job_from_row,
    load_done_ids,
    load_trace_rows,
)


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
    ap.add_argument("--layers", default="7,15,23,31")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens-per-batch", type=int, default=32768)
    ap.add_argument("--length-bucket-tokens", type=int, default=256)
    ap.add_argument("--pad-to-multiple-of", type=int, default=64)
    ap.add_argument("--empty-cache-every", type=int, default=1,
                    help="torch.cuda.empty_cache() every N batches (0 = never). Returns unused "
                         "reserved memory to the OS, i.e. hands it to other tenants -- set 0 "
                         "together with --reserve-gb when the GPU is shared and you want to keep "
                         "your headroom for the whole run.")
    ap.add_argument("--reserve-gb", type=float, default=0.0,
                    help="Claim this much GPU memory before loading the model, and hold it for the "
                         "whole run (like vLLM's gpu-memory-utilization). This is the process's "
                         "TOTAL footprint: ~15 GiB of weights plus activations. Measured peak on "
                         "the longest MuSiQue dev traces at the default --max-tokens-per-batch is "
                         "37 GiB, so 40 leaves a little headroom; lower --max-tokens-per-batch if "
                         "you need to fit in less. 0 = off.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not args.trace_file.is_file():
        sys.exit(f"Trace file not found: {args.trace_file}")

    layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    split_dir = args.out_dir or (HERE / "hidden_states_full_multilayer" / args.split)
    manifest_path = split_dir / "manifest.jsonl"

    if args.resume:
        check_resume_layers_match(manifest_path, layers)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    rows = load_trace_rows(args.trace_file, args.limit)
    done_ids = load_done_ids(manifest_path) if args.resume else set()
    todo_rows = [r for r in rows if str(r.get("id", "")).strip() not in done_ids]

    print("=" * 60)
    print("full-prefix, multi-layer hidden state extraction (error_propagation_probe)")
    print(f"  trace:  {args.trace_file}")
    print(f"  out:    {split_dir}")
    print(f"  model:  {args.model} layers={layers}")
    print(f"  device: {device} dtype={args.dtype}")
    print(f"  rows:   {len(todo_rows)} / {len(rows)} (skip {len(done_ids)} done)")
    print("=" * 60)

    if args.reserve_gb > 0 and device != "cpu":
        # Claim the whole footprint BEFORE loading weights: allocate a scratch tensor, then drop
        # the reference WITHOUT empty_cache() -- PyTorch's caching allocator keeps those blocks
        # reserved for this process and splits them for later allocations (weights included), so
        # a co-tenant cannot take the card out from under a long run, not even during the ~15s
        # weight load. Same idea as vLLM's gpu-memory-utilization. Pair with --empty-cache-every 0,
        # which otherwise hands the pool back every batch.
        n_elem = int(args.reserve_gb * (1024 ** 3) // 2)  # float16 = 2 bytes
        try:
            scratch = torch.empty(n_elem, dtype=torch.float16, device=device)
            del scratch
        except torch.cuda.OutOfMemoryError:
            free_b, total_b = torch.cuda.mem_get_info(torch.device(device))
            sys.exit(f"cannot reserve {args.reserve_gb} GiB on {device} "
                     f"({free_b / 1024 ** 3:.1f} GiB free) -- lower --reserve-gb or pick another GPU")
        free_b, total_b = torch.cuda.mem_get_info(torch.device(device))
        print(f"  reserved {args.reserve_gb:.1f} GiB up front "
              f"(card now shows {(total_b - free_b) / 1024 ** 3:.1f}/{total_b / 1024 ** 3:.1f} GiB used)"
              + ("  [WARNING: --empty-cache-every > 0 hands this back; pass 0]"
                 if args.empty_cache_every > 0 else ""))
        # so the peak printed at the end reflects what the run actually used, not the reservation
        torch.cuda.reset_peak_memory_stats(torch.device(device))

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

    (split_dir / "activations" / "pos").mkdir(parents=True, exist_ok=True)
    (split_dir / "activations" / "neg").mkdir(parents=True, exist_ok=True)

    ml_jobs: list[MLJob] = []
    for row in todo_rows:
        try:
            job = full_trace_job_from_row(row)
            ml_jobs.append(MLJob(job=job))
        except Exception as exc:
            rid = str(row.get("id", "")).strip()
            print(json.dumps({"error": str(exc), "id": rid}, ensure_ascii=False))

    manifest_mode = "a" if args.resume and manifest_path.exists() else "w"
    with open(manifest_path, manifest_mode, encoding="utf-8") as mf:
        stats = run_extract_and_save(
            ml_jobs, model=model, tokenizer=tokenizer, layers=layers,
            batch_size=args.batch_size, length_bucket_tokens=args.length_bucket_tokens,
            max_tokens_per_batch=args.max_tokens_per_batch, out_dir=split_dir,
            manifest_fp=mf, pad_to_multiple_of=args.pad_to_multiple_of,
            empty_cache_every=args.empty_cache_every,
        )

    if device != "cpu":
        peak = torch.cuda.max_memory_allocated(torch.device(device)) / 1024 ** 3
        reserved = torch.cuda.max_memory_reserved(torch.device(device)) / 1024 ** 3
        print(f"  peak GPU memory: {peak:.1f} GiB allocated / {reserved:.1f} GiB reserved "
              f"(weights + activations; size --reserve-gb against the reserved figure minus weights)")

    meta = {
        "model": args.model, "layers": layers, "dtype": args.dtype,
        "full_forward_all_prefixes": True, "n_ok": stats["n_ok"], "n_err": stats["n_err"],
        "trace_file": str(args.trace_file), "manifest": str(manifest_path),
    }
    (split_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"done ok={stats['n_ok']} err={stats['n_err']} -> {split_dir}")


if __name__ == "__main__":
    main()
