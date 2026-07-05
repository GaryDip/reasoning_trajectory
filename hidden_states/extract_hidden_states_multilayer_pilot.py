#!/usr/bin/env python3
"""
Pilot: multi-layer hidden-state extraction, to test whether an earlier/middle
transformer layer carries a stronger "does this evidence fit the chain"
signal than the production choice (last layer, 31).

Only one change vs. extract_hidden_states.py: output_hidden_states=True
already computes every layer in one forward pass; extract_hidden_states.py
only reads out layer 31 and discards the rest. This script reads out several
layers (--layers) from the same forward — no extra GPU time versus
single-layer extraction.

Still one forward per cumulative prefix, same as production — NOT one forward
per trace. An earlier version of this script tried to reuse a single forward
across all of a trace's prefixes (relying on causal masking: a token's hidden
state only depends on tokens at or before it). That reasoning has a hole
specific to this codebase: render_chat_text wraps *whatever text it's given*
as its own complete, closed chat turn, appending an end-of-turn token right
after it. Every prefix — short or long — gets its own end-of-turn token when
rendered on its own, matching how both this extraction script and the online
gate (run_retrieval_exp_wavefront.py) actually use it: at every hop, "what's
been said so far" is packaged as a finished turn and the model's reaction to
that closure is what's read out. Reusing one forward across prefixes instead
reads the hidden state while the model is mid-sequence with more text still
to come — that's read out at a token position with no end-of-turn immediately
after it, i.e. a materially different conditioning context, not merely a
faster way to compute the same number. So each prefix still needs its own
independent, end-of-turn-closed forward; only the multi-layer read-out is
free.

Train: stratified subsample by (K, pos/neg-hop) bucket — this is a probe-
comparison pilot, not the production gate, so it only needs enough rows per
bucket to fit a PCA+LR reasonably (not the full ~65k rows). Pooled gate
training pools each transition j across every K > j (fit_lr_gate_pooled.py),
so the deepest transition is fed *exclusively* by the largest-K traces (e.g.
only K=4 has a hop-4 transition at all, and MuSiQue only has ~2k such traces
to begin with — see traces/merged/musique/train.stats.json). A flat cap per
bucket would shrink that already-scarce bucket even further, so the largest K
present (`--full-k`, auto-detected by default) is sampled at 100% instead of
being capped; only the smaller, redundant K's are capped.

Dev: full split, used for evaluation later (not subsampled).

Memory: each trace's npz is written to disk and evicted from memory as soon as
all of its prefixes are filled (a trace's prefixes can land in different
length buckets since later hops are longer, so completion is tracked per
trace rather than assumed to happen within one batch) — earlier versions held
every trace in memory for the whole split before writing anything.
output_hidden_states=True also returns *every* one of the model's ~32 layers
at once regardless of how many you keep, so long sequences at a fixed
example-count batch size can spike GPU memory; --max-tokens-per-batch shrinks
the effective batch size for buckets of long sequences to keep that bounded.

Output mirrors hidden_states/{split}/ layout under hidden_states/pilot_multilayer/{split}/,
with hidden states for all requested layers stacked in one array per trace so a
downstream fit script (gate/compare_layers_pilot.py) can slice out any single
layer or concatenate several.

Usage:
  python extract_hidden_states_multilayer_pilot.py --layers 7,15,23,31
  python extract_hidden_states_multilayer_pilot.py --layers 15,31 --per-bucket-cap 800 --dev-limit 2000
  python extract_hidden_states_multilayer_pilot.py --full-k 3,4 --per-bucket-cap 1500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from extract_hidden_states import (
    TraceJob,
    bucket_works_by_length,
    load_done_ids,
    load_trace_rows,
    prefix_token_len,
    render_chat_text,
    safe_file_stem,
)
from trace_parse import first_wrong_hop_from_row


# ---------------------------------------------------------------------------
# Stratified train subsample
# ---------------------------------------------------------------------------

def bucket_key(row: dict[str, Any]) -> tuple[int, str]:
    k = int(row.get("K") or 0)
    if row.get("trace_type") == "error":
        wh = first_wrong_hop_from_row(row)
        return (k, f"neg_h{wh}")
    return (k, "pos")


def stratified_sample(
    rows: list[dict[str, Any]],
    *,
    per_bucket_cap: int,
    full_ks: set[int],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """
    Stratified subsample by (K, pos/neg-hop) bucket.

    Pooled gate training pools each semantic transition j across every K > j
    (see gate/fit_lr_gate_pooled.py). Transition j is therefore only as rich as
    its *smallest* contributing K — and the deepest transition (j = max_K - 1)
    is fed exclusively by the rarest, largest-K traces (e.g. only K=4 traces
    carry a hop-4 transition at all). A flat cap applied identically to every
    K bucket disproportionately starves that already-scarce bucket further.
    So `full_ks` (default: the largest K present) is sampled at 100% — no cap
    — while smaller K's, which are inherently more abundant *and* already feed
    multiple pooled transitions, are capped to keep the pilot's total size
    manageable.
    """
    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[bucket_key(r)].append(r)
    out: list[dict[str, Any]] = []
    report: dict[str, dict[str, int]] = {}
    for key, items in sorted(buckets.items()):
        k = key[0]
        if k in full_ks or len(items) <= per_bucket_cap:
            take = items
        else:
            take = rng.sample(items, per_bucket_cap)
        out.extend(take)
        report[f"K{key[0]}_{key[1]}"] = {
            "pool": len(items), "taken": len(take), "full": k in full_ks,
        }
    rng.shuffle(out)
    return out, report


# ---------------------------------------------------------------------------
# Multi-layer, per-prefix extraction (each prefix its own independent,
# end-of-turn-closed forward — matches production semantics exactly).
# ---------------------------------------------------------------------------

@dataclass
class MLJob:
    job: TraceJob
    hidden_by_layer: dict[int, list[np.ndarray | None]] = field(default_factory=dict)
    n_filled: int = 0  # how many of job.n_take prefixes have all layers filled


@dataclass
class PrefixWork:
    mlj: MLJob
    pfx_idx: int
    text: str
    token_len: int = 0


def bucket_batch_size(bucket: list[PrefixWork], *, batch_size: int, max_tokens_per_batch: int) -> int:
    """Shrink the batch size for buckets of long sequences so batch_size *
    seq_len (padded) stays under a token budget — output_hidden_states=True
    materializes *every* layer's (batch, seq_len, hidden) tensor at once
    (33 layers for Llama-3.1-8B, not just the few we keep), so a fixed
    example-count batch size that's fine for short prefixes can still OOM on
    long ones."""
    if not bucket or max_tokens_per_batch <= 0:
        return batch_size
    longest = max(w.token_len for w in bucket)
    if longest <= 0:
        return batch_size
    return max(1, min(batch_size, max_tokens_per_batch // longest))


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_ml_job_npz(mlj: MLJob, layers: list[int], out_dir: Path) -> Path:
    job = mlj.job
    n_take = job.n_take
    stacked = np.stack(
        [np.stack(mlj.hidden_by_layer[layer], axis=0) for layer in layers], axis=0
    )  # (n_layers, n_take, hidden_dim)
    wh = first_wrong_hop_from_row(job.row) if job.subdir == "neg" else None
    wrong_hops = job.row.get("wrong_hops") or ([] if wh is None else [wh])

    stem = safe_file_stem(f"{job.rid}_{job.subdir}" + (f"_wh{wh}" if wh is not None else ""))
    out_path = out_dir / "activations" / job.subdir / f"{stem}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        hidden=stacked,
        layers=np.array(layers, dtype=np.int32),
        trace_pos=np.arange(1, n_take + 1, dtype=np.int32),
        hidden_size=np.array([stacked.shape[-1]], dtype=np.int32),
        K=np.array([job.k], dtype=np.int32),
        example_id=np.array(job.rid, dtype=object),
        source_id=np.array(str(job.row.get("source_id") or job.rid), dtype=object),
        wrong_hops=np.array(wrong_hops, dtype=np.int32),
        first_wrong_hop=np.array([-1 if wh is None else wh], dtype=np.int32),
        label=np.array(job.label, dtype=object),
        trace_type=np.array("correct" if job.subdir == "pos" else "error", dtype=object),
    )
    return out_path


def write_manifest_row(mf, mlj: MLJob, layers: list[int], out_path: Path, out_dir: Path) -> None:
    job = mlj.job
    wh = first_wrong_hop_from_row(job.row)
    mf.write(json.dumps({
        "id": job.rid,
        "label": job.label,
        "trace_type": job.row.get("trace_type"),
        "K": job.k,
        "wrong_hops": job.row.get("wrong_hops") or ([] if wh is None else [wh]),
        "first_wrong_hop": wh,
        "num_prefixes_saved": job.n_take,
        "layers": layers,
        "path": str(out_path.relative_to(out_dir)),
        "source_id": job.row.get("source_id"),
    }, ensure_ascii=False) + "\n")
    mf.flush()


def run_extract_and_save(
    ml_jobs: list[MLJob],
    *,
    model,
    tokenizer,
    layers: list[int],
    batch_size: int,
    length_bucket_tokens: int,
    max_tokens_per_batch: int,
    out_dir: Path,
    manifest_fp,
    pad_to_multiple_of: int = 64,
    empty_cache_every: int = 1,
) -> dict[str, int]:
    """
    Runs the per-prefix multi-layer forward and writes+evicts each trace's
    npz as soon as all of its prefixes are filled, instead of holding every
    trace's hidden states in memory for the whole split before writing
    anything. A trace's prefixes can land in different length buckets (later
    hops are longer), so completion is tracked per job via `n_filled` rather
    than assuming a job finishes within one batch.

    Two things beyond that to keep GPU memory from creeping up over the
    course of a long run with many differently-shaped batches:

    - `padding=True` alone pads each chunk to *that chunk's own* max length,
      which is rarely the same number twice. output_hidden_states=True
      materializes every one of the model's ~32 layers at that exact shape,
      so PyTorch's caching allocator ends up minting a new memory block for
      almost every batch instead of reusing one — the reserved pool nvidia-smi
      reports grows continuously and never comes back down, since freed
      blocks of one shape can't serve a request of a different shape.
      `pad_to_multiple_of` rounds every batch's padded length up to a fixed
      grid, so far more batches land on the same handful of shapes and the
      allocator can actually reuse its blocks instead of piling up new ones.
    - `torch.cuda.empty_cache()` every `empty_cache_every` batches forces
      whatever's sitting unused in the reserved pool back to the OS-visible
      free pool, trading a bit of allocation overhead later for a memory
      curve that doesn't just keep climbing.
    """
    import torch

    n_ok = n_err = 0
    if not ml_jobs:
        return {"n_ok": 0, "n_err": 0}

    works: list[PrefixWork] = []
    for mlj in ml_jobs:
        for i, pfx in enumerate(mlj.job.prefixes[: mlj.job.n_take]):
            works.append(PrefixWork(mlj, i, pfx, token_len=prefix_token_len(tokenizer, pfx)))

    buckets = bucket_works_by_length(works, max_spread=length_bucket_tokens)
    is_cuda = getattr(model.device, "type", str(model.device)) == "cuda"
    pbar = tqdm(total=len(ml_jobs), desc="extract+save", unit="ex")
    n_batches = 0
    for bucket in buckets:
        eff_bs = bucket_batch_size(
            bucket, batch_size=batch_size, max_tokens_per_batch=max_tokens_per_batch
        )
        for start in range(0, len(bucket), eff_bs):
            chunk = bucket[start : start + eff_bs]
            rendered = [render_chat_text(tokenizer, w.text) for w in chunk]
            batch = tokenizer(
                rendered, return_tensors="pt", padding=True,
                pad_to_multiple_of=pad_to_multiple_of or None,
            )
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hs = out.hidden_states
            last_indices = (
                attention_mask.size(1) - 1 - torch.flip(attention_mask, dims=[1]).argmax(dim=1)
            )
            batch_indices = torch.arange(len(chunk), device=model.device)
            for layer in layers:
                layer_h = hs[layer + 1]
                gathered = (
                    layer_h[batch_indices, last_indices].float().cpu().numpy().astype(np.float32)
                )
                for w, vec in zip(chunk, gathered):
                    vecs = w.mlj.hidden_by_layer.setdefault(layer, [None] * w.mlj.job.n_take)
                    vecs[w.pfx_idx] = vec
            del hs, out, input_ids, attention_mask, batch, layer_h, gathered, last_indices, batch_indices

            n_batches += 1
            if is_cuda and empty_cache_every > 0 and n_batches % empty_cache_every == 0:
                torch.cuda.empty_cache()

            # Flush + evict any job whose prefixes are now all filled, so we
            # never hold more than "in-flight" jobs' hidden states at once.
            # A chunk can contain more than one prefix of the same job, so
            # count how many of *this* job's prefixes this chunk just filled
            # rather than crediting each touched job a flat +1.
            counts: dict[int, list] = {}
            for w in chunk:
                entry = counts.setdefault(id(w.mlj), [w.mlj, 0])
                entry[1] += 1
            for mlj, n_new in counts.values():
                mlj.n_filled += n_new
                if mlj.n_filled < mlj.job.n_take:
                    continue
                job = mlj.job
                try:
                    missing = [
                        layer for layer in layers
                        if any(v is None for v in mlj.hidden_by_layer.get(layer, [None]))
                    ]
                    if missing:
                        raise ValueError(f"missing hidden vectors for layers {missing}")
                    out_path = save_ml_job_npz(mlj, layers, out_dir)
                    write_manifest_row(manifest_fp, mlj, layers, out_path, out_dir)
                    n_ok += 1
                except Exception as exc:
                    n_err += 1
                    tqdm.write(json.dumps({"error": str(exc), "id": job.rid}, ensure_ascii=False))
                finally:
                    mlj.hidden_by_layer.clear()  # free the in-memory arrays now that it's on disk
                    pbar.update(1)
    pbar.close()
    return {"n_ok": n_ok, "n_err": n_err}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_resume_layers_match(manifest_path: Path, layers: list[int]) -> None:
    """A resumed run must extract the same layers as whatever's already on
    disk — otherwise some traces in the split would end up with 4 layers and
    others (the resumed ones) with a different set, silently corrupting the
    npz format compare_layers_pilot.py expects."""
    if not manifest_path.is_file():
        return
    with open(manifest_path, encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        return
    prev_layers = json.loads(first_line).get("layers")
    if prev_layers is not None and sorted(prev_layers) != sorted(layers):
        sys.exit(
            f"--resume: existing manifest at {manifest_path} was extracted with "
            f"layers={sorted(prev_layers)}, but this run asked for layers={sorted(layers)}. "
            "Use the same --layers to resume, or a fresh --out-dir to start over."
        )


def extract_split(
    rows: list[dict[str, Any]],
    *,
    split: str,
    out_dir: Path,
    model,
    tokenizer,
    layers: list[int],
    batch_size: int,
    length_bucket_tokens: int,
    max_tokens_per_batch: int,
    pad_to_multiple_of: int,
    empty_cache_every: int,
    resume: bool,
) -> dict[str, int]:
    activ_root = out_dir / "activations"
    (activ_root / "pos").mkdir(parents=True, exist_ok=True)
    (activ_root / "neg").mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    if resume:
        check_resume_layers_match(manifest_path, layers)
    done_ids = load_done_ids(manifest_path) if resume else set()
    todo_rows = [r for r in rows if str(r.get("id", "")).strip() not in done_ids]
    print(f"[{split}] resume: skipping {len(done_ids)} already-done, {len(todo_rows)}/{len(rows)} left")

    n_parse_err = 0
    ml_jobs: list[MLJob] = []

    for row in tqdm(todo_rows, desc=f"{split}: parse", unit="ex"):
        try:
            job = TraceJob.from_row(row)
        except Exception as exc:
            n_parse_err += 1
            tqdm.write(json.dumps({"error": str(exc), "id": str(row.get("id", ""))}, ensure_ascii=False))
            continue
        ml_jobs.append(MLJob(job=job))

    manifest_mode = "a" if resume and manifest_path.exists() else "w"
    with open(manifest_path, manifest_mode, encoding="utf-8") as mf:
        stats = run_extract_and_save(
            ml_jobs, model=model, tokenizer=tokenizer, layers=layers,
            batch_size=batch_size, length_bucket_tokens=length_bucket_tokens,
            max_tokens_per_batch=max_tokens_per_batch, out_dir=out_dir, manifest_fp=mf,
            pad_to_multiple_of=pad_to_multiple_of, empty_cache_every=empty_cache_every,
        )

    return {"n_ok": stats["n_ok"], "n_err": stats["n_err"] + n_parse_err}


def main() -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"Install torch + transformers: {exc}") from exc

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="musique")
    ap.add_argument("--layers", default="7,15,23,31",
                    help="Comma-separated 0-based layer indices to extract (last layer=31 for Llama-3.1-8B).")
    ap.add_argument("--per-bucket-cap", type=int, default=1500,
                    help="Max train rows sampled per (K, pos/neg-hop) bucket, for K not in --full-k.")
    ap.add_argument("--full-k", default=None,
                    help="Comma-separated K values to sample at 100%% (no cap). "
                         "Default: auto = the largest K present in the data, since pooled "
                         "gate transitions are pooled across K and the deepest transition "
                         "(hop max_K) is only ever fed by that largest-K bucket already.")
    ap.add_argument("--dev-limit", type=int, default=0, help="0 = full dev split.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--length-bucket-tokens", type=int, default=256)
    ap.add_argument("--max-tokens-per-batch", type=int, default=12000,
                    help="Shrinks the effective batch size for buckets of long sequences so "
                         "batch_size * seq_len stays under this budget (output_hidden_states=True "
                         "materializes every one of the model's ~32 layers at once, not just the "
                         "few requested, so long sequences at full --batch-size can OOM). 0 = off.")
    ap.add_argument("--pad-to-multiple-of", type=int, default=64,
                    help="Round every batch's padded length up to a multiple of this, so many "
                         "batches share the same shape and PyTorch's allocator can reuse memory "
                         "blocks instead of minting a new one per batch (GPU memory otherwise "
                         "creeps up all run and never comes back down). 0 = off.")
    ap.add_argument("--empty-cache-every", type=int, default=1,
                    help="Call torch.cuda.empty_cache() every N batches to return unused reserved "
                         "memory to the OS-visible free pool. 0 = never (fastest, but memory "
                         "reported by nvidia-smi will only grow over the run).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip ids already present in an existing manifest.jsonl for each split "
                         "(train_pilot / dev), instead of overwriting from scratch. Requires the "
                         "same --seed/--per-bucket-cap/--full-k as the original run, since the "
                         "train subsample is only reproducible when those match, and the same "
                         "--layers, since a resumed run can't extract a different layer set into "
                         "the same file.")
    args = ap.parse_args()

    layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    out_root = args.out_dir or (PROJECT_ROOT / "hidden_states" / "pilot_multilayer")

    train_path = PROJECT_ROOT / "traces" / "merged" / args.dataset / "train.jsonl"
    dev_path = PROJECT_ROOT / "traces" / "merged" / args.dataset / "dev.jsonl"
    if not train_path.is_file() or not dev_path.is_file():
        sys.exit(f"Missing merged trace file(s): {train_path} / {dev_path}")

    rng = random.Random(args.seed)
    train_rows_all = load_trace_rows(train_path, limit=0)

    if args.full_k:
        full_ks = {int(x) for x in args.full_k.split(",") if x.strip()}
    else:
        full_ks = {max(int(r.get("K") or 0) for r in train_rows_all)}

    train_rows, bucket_report = stratified_sample(
        train_rows_all, per_bucket_cap=args.per_bucket_cap, full_ks=full_ks, rng=rng
    )
    dev_rows = load_trace_rows(dev_path, limit=args.dev_limit)

    print("=" * 60)
    print("multilayer pilot extraction")
    print(f"  layers:   {layers}")
    print(f"  full_ks:  {sorted(full_ks)} (sampled at 100%, no cap)")
    print(f"  train:    {len(train_rows)} sampled / {len(train_rows_all)} available "
          f"(cap={args.per_bucket_cap}/bucket for K not in full_ks)")
    print(f"  dev:      {len(dev_rows)}")
    print(f"  out:      {out_root}")
    print("=" * 60)
    for key, v in sorted(bucket_report.items()):
        tag = "FULL" if v["full"] else f"cap={args.per_bucket_cap}"
        print(f"  {key:<14} pool={v['pool']:<6} taken={v['taken']:<6} ({tag})")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
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

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is not None and any(layer >= num_layers for layer in layers):
        sys.exit(f"--layers has an index >= num_hidden_layers ({num_layers}): {layers}")

    stats = {}
    for split, rows in (("train_pilot", train_rows), ("dev", dev_rows)):
        stats[split] = extract_split(
            rows,
            split=split,
            out_dir=out_root / split,
            model=model,
            tokenizer=tokenizer,
            layers=layers,
            batch_size=args.batch_size,
            length_bucket_tokens=args.length_bucket_tokens,
            max_tokens_per_batch=args.max_tokens_per_batch,
            pad_to_multiple_of=args.pad_to_multiple_of,
            empty_cache_every=args.empty_cache_every,
            resume=args.resume,
        )

    meta = {
        "model": args.model,
        "layers": layers,
        "dtype": args.dtype,
        "per_bucket_cap": args.per_bucket_cap,
        "full_ks": sorted(full_ks),
        "seed": args.seed,
        "train_source": str(train_path),
        "dev_source": str(dev_path),
        "bucket_report": bucket_report,
        "stats": stats,
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\ndone.")
    for split, s in stats.items():
        print(f"  {split}: ok={s['n_ok']} err={s['n_err']}")
    print(f"meta -> {out_root / 'run_meta.json'}")


if __name__ == "__main__":
    main()
