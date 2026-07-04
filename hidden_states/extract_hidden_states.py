#!/usr/bin/env python3
"""
Extract last-token hidden states from merged trace jsonl.

Per trace: full forward on each cumulative prefix (matches training distribution).
Batched across prefixes (left-pad); no incremental KV.

Input:  traces/merged/{dataset}/{split}.jsonl
Output: hidden_states/{split}/activations/{pos,neg}/*.npz + manifest.jsonl

Usage:
  python extract_hidden_states.py --split train --trace-file ../traces/merged/musique/train.jsonl
  python extract_hidden_states.py --device cuda:0 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from trace_parse import (
    cumulative_prefix_strings,
    first_wrong_hop_from_row,
    parse_trace_structure,
)


def safe_file_stem(s: str, max_len: int = 200) -> str:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:12]
    t = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:max_len].strip("_")
    return f"{t}__{h}" if t else h


def render_chat_text(tokenizer: Any, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=False,
    )


def encode_chat(tokenizer: Any, user_text: str) -> Any:
    """Chat-template encode; returns 1D input_ids."""
    return encode_chat_batch(tokenizer, user_text)[0]


def encode_chat_batch(tokenizer: Any, user_text: str) -> Any:
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_tensors="pt",
    )


@dataclass
class TraceJob:
    row: dict[str, Any]
    rid: str
    label: str
    subdir: str
    k: int
    n_take: int
    prefixes: list[str]
    hiddens: list[np.ndarray] = field(default_factory=list)
    hidden_by_idx: dict[int, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> TraceJob:
        trace = (row.get("reasoning_trace") or "").strip()
        if not trace:
            raise ValueError("empty reasoning_trace")
        q, blocks, _ans = parse_trace_structure(trace)
        prefixes = cumulative_prefix_strings(q, blocks)
        k = len(blocks)
        trace_type = str(row.get("trace_type", "correct"))
        if trace_type == "error":
            wh = first_wrong_hop_from_row(row)
            if wh is None:
                raise ValueError("error trace missing first wrong hop")
            n_take = wh + 1
            label = "negative"
            subdir = "neg"
        else:
            n_take = len(prefixes)
            label = "positive"
            subdir = "pos"
        if n_take > len(prefixes):
            raise ValueError(f"wrong_hop too large for {len(prefixes)} prefixes")
        rid = str(row.get("id", "")).strip()
        row_k = row.get("K")
        k_out = int(row_k) if row_k is not None else k
        return cls(
            row=row,
            rid=rid,
            label=label,
            subdir=subdir,
            k=k_out,
            n_take=n_take,
            prefixes=prefixes,
        )


@dataclass
class PrefixWork:
    job: TraceJob
    pfx_idx: int
    text: str
    token_len: int = 0


def prefix_token_len(tokenizer: Any, text: str) -> int:
    rendered = render_chat_text(tokenizer, text)
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def bucket_works_by_length(
    works: list[PrefixWork],
    *,
    max_spread: int,
) -> list[list[PrefixWork]]:
    """Sort by token length; group so len - bucket_min <= max_spread."""
    if not works:
        return []
    if max_spread <= 0:
        return [works]
    sorted_works = sorted(works, key=lambda w: w.token_len)
    buckets: list[list[PrefixWork]] = []
    cur: list[PrefixWork] = []
    cur_min: int | None = None
    for w in sorted_works:
        if not cur or (cur_min is not None and w.token_len - cur_min <= max_spread):
            cur.append(w)
            if cur_min is None:
                cur_min = w.token_len
        else:
            buckets.append(cur)
            cur = [w]
            cur_min = w.token_len
    if cur:
        buckets.append(cur)
    return buckets

class HiddenExtractor:
    """Full forward per cumulative prefix; batched with left padding."""

    def __init__(self, model: Any, tokenizer: Any, *, layer: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.device = model.device

    def batch_last_hidden(self, texts: list[str], *, batch_size: int) -> list[np.ndarray]:
        import torch

        if not texts:
            return []
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        out: list[np.ndarray] = []
        rendered = [render_chat_text(self.tokenizer, t) for t in texts]
        for start in range(0, len(rendered), batch_size):
            chunk = rendered[start : start + batch_size]
            batch = self.tokenizer(chunk, return_tensors="pt", padding=True)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            with torch.inference_mode():
                model_out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hs = model_out.hidden_states
            if hs is None or self.layer + 1 >= len(hs):
                raise ValueError(f"bad hidden_states for layer={self.layer}")
            layer_h = hs[self.layer + 1]
            last_indices = (
                attention_mask.size(1)
                - 1
                - torch.flip(attention_mask, dims=[1]).argmax(dim=1)
            )
            batch_indices = torch.arange(layer_h.size(0), device=self.device)
            hidden = (
                layer_h[batch_indices, last_indices]
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            out.extend(hidden[i] for i in range(hidden.shape[0]))
        return out

    def run_work(
        self,
        works: list[PrefixWork],
        *,
        batch_size: int,
        length_bucket_tokens: int = 256,
    ) -> None:
        if not works:
            return
        buckets = bucket_works_by_length(works, max_spread=length_bucket_tokens)
        for bucket in buckets:
            vecs = self.batch_last_hidden([w.text for w in bucket], batch_size=batch_size)
            for work, vec in zip(bucket, vecs):
                work.job.hidden_by_idx[work.pfx_idx] = vec

    def finalize_job(self, job: TraceJob) -> None:
        job.hiddens = [job.hidden_by_idx[i] for i in range(job.n_take)]


def save_job_npz(job: TraceJob, out_dir: Path) -> Path:
    hidden = np.stack(job.hiddens, axis=0)
    n_take = hidden.shape[0]
    pos_indices = np.arange(1, n_take + 1, dtype=np.int32)
    wh = first_wrong_hop_from_row(job.row) if job.subdir == "neg" else None
    wrong_hops = job.row.get("wrong_hops") or ([] if wh is None else [wh])

    stem = safe_file_stem(
        f"{job.rid}_{job.subdir}" + (f"_wh{wh}" if wh is not None else "")
    )
    out_path = out_dir / "activations" / job.subdir / f"{stem}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        hidden=hidden,
        trace_pos=pos_indices,
        hidden_size=np.array([hidden.shape[-1]], dtype=np.int32),
        K=np.array([job.k], dtype=np.int32),
        example_id=np.array(job.rid, dtype=object),
        source_id=np.array(str(job.row.get("source_id") or job.rid), dtype=object),
        wrong_hops=np.array(wrong_hops, dtype=np.int32),
        wrong_evidence_at_hop=np.array([-1 if wh is None else wh], dtype=np.int32),
        first_wrong_hop=np.array([-1 if wh is None else wh], dtype=np.int32),
        label=np.array(job.label, dtype=object),
        trace_type=np.array(
            "correct" if job.subdir == "pos" else "error", dtype=object
        ),
    )
    return out_path


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

    ap = argparse.ArgumentParser(description="Extract hidden states (batched full forward).")
    ap.add_argument("--split", choices=("train", "dev"), default="train")
    ap.add_argument("--trace-file", type=Path, default=None)
    ap.add_argument("--trace-source", choices=("gold", "counterfactual", "merged"), default="merged")
    ap.add_argument("--dataset", default="musique")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Prefix forward batch size (left-padded)")
    ap.add_argument("--trace-chunk-size", type=int, default=32,
                    help="Traces per chunk before flattening prefixes into batches")
    ap.add_argument("--length-bucket-tokens", type=int, default=256,
                    help="Group prefixes into length buckets (max spread in tokens); 0=off")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="Skip ids in existing manifest")
    args = ap.parse_args()

    if args.trace_file is None:
        args.trace_file = (
            PROJECT_ROOT / "traces" / args.trace_source / args.dataset / f"{args.split}.jsonl"
        )
    split_dir = args.out_dir or (PROJECT_ROOT / "hidden_states" / args.split)
    activ_root = split_dir / "activations"
    manifest_path = split_dir / "manifest.jsonl"

    if not args.trace_file.is_file():
        sys.exit(f"Trace file not found: {args.trace_file}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    rows = load_trace_rows(args.trace_file, args.limit)
    done_ids: set[str] = load_done_ids(manifest_path) if args.resume else set()
    todo_rows = [r for r in rows if str(r.get("id", "")).strip() not in done_ids]

    print("=" * 60)
    print("hidden state extraction (full forward per prefix)")
    print(f"  trace:  {args.trace_file}")
    print(f"  out:    {split_dir}")
    print(f"  model:  {args.model} layer={args.layer}")
    print(f"  device: {device} dtype={args.dtype}")
    print(f"  batch:  prefix_batch={args.batch_size} trace_chunk={args.trace_chunk_size} "
          f"len_bucket={args.length_bucket_tokens}")
    print(f"  rows:   {len(todo_rows)} / {len(rows)} (skip {len(done_ids)} done)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dev_map = {"": device} if device != "cpu" else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map=dev_map,
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
                    job = TraceJob.from_row(row)
                    jobs.append(job)
                    row_by_job.append((row, job))
                    for i, pfx in enumerate(job.prefixes[: job.n_take]):
                        works.append(PrefixWork(
                            job, i, pfx,
                            token_len=prefix_token_len(tokenizer, pfx),
                        ))
                except Exception as exc:
                    n_err += 1
                    pbar.update(1)
                    tqdm.write(json.dumps({"error": str(exc), "id": rid}, ensure_ascii=False))

            if works:
                try:
                    extractor.run_work(
                        works,
                        batch_size=args.batch_size,
                        length_bucket_tokens=args.length_bucket_tokens,
                    )
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
                    wh = first_wrong_hop_from_row(job.row)
                    mf.write(json.dumps({
                        "id": job.rid,
                        "label": job.label,
                        "trace_type": job.row.get("trace_type"),
                        "K": job.k,
                        "wrong_hops": job.row.get("wrong_hops") or ([] if wh is None else [wh]),
                        "first_wrong_hop": wh,
                        "wrong_evidence_at_hop": wh,
                        "num_hops": job.k,
                        "num_prefixes_saved": len(job.hiddens),
                        "trace_pos_columns": list(range(1, len(job.hiddens) + 1)),
                        "model_layer_0based": args.layer,
                        "path": str(out_path.relative_to(split_dir)),
                        "source_id": job.row.get("source_id"),
                    }, ensure_ascii=False) + "\n")
                    n_ok += 1
                except Exception as exc:
                    n_err += 1
                    tqdm.write(json.dumps({"error": str(exc), "id": job.rid}, ensure_ascii=False))
                pbar.update(1)
        pbar.close()

    meta = {
        "model": args.model,
        "layer_0based": args.layer,
        "dtype": args.dtype,
        "incremental_kv": False,
        "full_forward_per_prefix": True,
        "prefix_batch_size": args.batch_size,
        "trace_chunk_size": args.trace_chunk_size,
        "length_bucket_tokens": args.length_bucket_tokens,
        "cross_prefix_batch": True,
        "n_ok": n_ok,
        "n_err": n_err,
        "trace_file": str(args.trace_file),
        "manifest": str(manifest_path),
    }
    (split_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"done ok={n_ok} err={n_err} → {split_dir}")


if __name__ == "__main__":
    main()
