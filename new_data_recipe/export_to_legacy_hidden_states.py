#!/usr/bin/env python3
"""
Step 11 (see update_doc/0907/0907update.md section 9): adapter that makes traces_v2 +
hidden_states_v2 loadable by the EXISTING, unmodified gate/fit_lr_gate_pooled.py -- confirms
"only stage 2 changes, stage 3/4 don't" by construction, rather than editing those scripts.

gate/fit_lr_gate.py::read_hidden_from_npz(z, layer=None) already accepts `hidden` as a plain
(T, hidden_dim) array with no other required keys, and read_K_from_npz falls back to
hidden.shape[0]-1 when no "K" key is present -- so hidden_states_v2's npz files are ALREADY
byte-compatible. The only two things collect_split() needs that hidden_states_v2 doesn't
provide are: (1) `wrong_hops` inside the npz (parse_wrong_hops reads it straight from `z` when
no manifest row is given), and (2) the pos/neg SUBDIRECTORY split (collect_split infers
trace_type from which subdir a file lives in, not from any field). This script adds both by
copying (not moving) each trace's hidden_states_v2 npz -- plus a `wrong_hops` array -- into
  {legacy-hs-root}/{split}/activations/{pos|neg}/{trace_id}.npz

Usage:
  python export_to_legacy_hidden_states.py --in-dir output_small_test --legacy-hs-root legacy_hidden_states_test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, required=True, help="output of build_dataset.py")
    ap.add_argument("--dataset", default="musique")
    ap.add_argument("--split", default="train")
    ap.add_argument("--legacy-hs-root", type=Path, required=True)
    args = ap.parse_args()

    traces_path = args.in_dir / "traces_v2" / args.dataset / f"{args.split}.jsonl"
    hs_dir = args.in_dir / "hidden_states_v2" / args.dataset / args.split
    pos_dir = args.legacy_hs_root / args.split / "activations" / "pos"
    neg_dir = args.legacy_hs_root / args.split / "activations" / "neg"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    n_pos = n_neg = n_missing = 0
    with traces_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            trace = json.loads(line)
            src_npz = hs_dir / f"{trace['trace_id']}.npz"
            if not src_npz.exists():
                n_missing += 1
                continue
            z = np.load(src_npz, allow_pickle=True)
            wrong_hops = [h["hop"] for h in trace["hops"] if not h["is_correct"]]
            out_dir = pos_dir if trace["all_correct"] else neg_dir
            out_path = out_dir / f"{trace['trace_id']}.npz"
            np.savez_compressed(
                out_path,
                hidden=z["hidden"],
                K=np.array([trace["K"]], dtype=np.int32),
                example_id=np.array(trace["trace_id"], dtype=object),
                source_id=np.array(trace["case_id"], dtype=object),
                wrong_hops=np.array(wrong_hops, dtype=np.int32),
                first_wrong_hop=np.array([wrong_hops[0] if wrong_hops else -1], dtype=np.int32),
            )
            if trace["all_correct"]:
                n_pos += 1
            else:
                n_neg += 1

    print(f"pos: {n_pos} -> {pos_dir}")
    print(f"neg: {n_neg} -> {neg_dir}")
    if n_missing:
        print(f"[warn] {n_missing} traces had no matching hidden_states_v2 npz, skipped")


if __name__ == "__main__":
    main()
