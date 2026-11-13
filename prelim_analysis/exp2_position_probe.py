#!/usr/bin/env python3
"""
Experiment 2 -- how much of the prefix index is linearly readable from the state?

Original exploration: one linear probe per K over prefix indices 1..K+1, ~100% accuracy
(0.9967 / 1.0000 / 0.9970), read as evidence for "step-specific geometry".

What that measurement can and cannot show. The probe reads hidden states, never text, so a high
accuracy is a genuine statement about the representation: the layer-15 last-token state linearly
encodes the prefix index. But it does not separate two hypotheses, because BOTH predict ~100%:

  H1  step-specific reasoning geometry -- different hops occupy different regions because the
      reasoning content differs;
  H2  a surface counter -- the state encodes "how many numbered blocks precede me", which is
      lexically determined: the prefix literally contains "Step 1:", "Step 2:", ... and the index
      is a deterministic function of that text.

Separating H1 from H2 needs a counterfactual *input* (strip or homogenise the "Step N:" markers)
and therefore a fresh extraction; it is not possible with the stored states. This script
therefore (a) reproduces the probe and reports it as "the index is linearly decodable", (b) adds
the length controls the plan asks for, and (c) adds a probe whose target is NOT present in the
text as a surface cue: predicting the chain length K from the state of the question-only prefix.

Zero GPU.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict

import numpy as np

import common

sys.path.insert(0, str(common.ROOT / "hidden_states"))
from trace_parse import cumulative_prefix_strings, parse_trace_structure  # noqa: E402

SPLIT_FILE = {"train_pilot": "train", "dev": "dev"}


def trace_texts(split: str) -> dict[str, str]:
    f = common.ROOT / "traces" / "merged" / "musique" / f"{SPLIT_FILE[split]}.jsonl"
    return {json.loads(l)["id"]: json.loads(l)["reasoning_trace"] for l in f.open() if l.strip()}


def gather(split: str, per_K: int, seed: int, want_tokens: bool, tok=None):
    """Return per-K dict of (X, prefix_index, token_len, source_id)."""
    rng = random.Random(seed)
    texts = trace_texts(split) if want_tokens else {}
    by_K = defaultdict(list)
    for r in common.manifest(split):
        if r["trace_type"] == "correct":
            by_K[int(r["K"])].append(r)
    out = {}
    for K, rows in sorted(by_K.items()):
        rng.shuffle(rows)
        rows = rows[:per_K]
        X, pos, tl, sid = [], [], [], []
        for r in rows:
            h, tp = common.load_states(r, split)
            X.append(h); pos.append(tp); sid += [r["source_id"]] * len(tp)
            if want_tokens:
                q, blocks, _ = parse_trace_structure(texts[r["id"]])
                pfx = cumulative_prefix_strings(q, blocks)
                tl += [len(tok(p)["input_ids"]) for p in pfx[: len(tp)]]
        out[K] = (np.concatenate(X), np.concatenate(pos),
                  np.array(tl) if want_tokens else None, np.array(sid))
        print(f"  {split} K={K}: {len(rows)} chains -> {out[K][0].shape[0]} prefixes")
    return out


def fit_probe(Xtr, ytr, Xte, yte, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=C, max_iter=2000, random_state=0)
    lr.fit(sc.transform(Xtr), ytr)
    pred = lr.predict(sc.transform(Xte))
    return float((pred == yte).mean()), confusion_matrix(yte, pred, labels=sorted(set(yte)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-K-train", type=int, default=4000)
    ap.add_argument("--per-K-test", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    print("Loading train_pilot ...")
    tr = gather("train_pilot", args.per_K_train, args.seed, True, tok)
    print("Loading dev ...")
    te = gather("dev", args.per_K_test, args.seed, True, tok)

    res = {"layer": common.LAYER, "train_split": "train_pilot", "test_split": "dev",
           "note": "probe trained on train, evaluated on dev; dev is never used to pick anything"}
    plt = common.setup_mpl()

    # ---------- (1) per-K prefix-index probe, + length controls ----------
    Ks = sorted(tr)
    fig, axes = plt.subplots(1, len(Ks), figsize=(common.FIG_W_CM * common.CM, 5.2 * common.CM))
    for ax, K in zip(np.atleast_1d(axes), Ks):
        Xtr, ytr, ltr, _ = tr[K]
        Xte, yte, lte, _ = te[K]
        acc, cm = fit_probe(Xtr, ytr, Xte, yte)
        acc_len, _ = fit_probe(ltr.reshape(-1, 1), ytr, lte.reshape(-1, 1), yte)
        chance = max(np.mean(yte == c) for c in np.unique(yte))
        res.setdefault("prefix_index_probe", {})[str(K)] = {
            "hidden_state_acc": round(acc, 4), "token_length_only_acc": round(acc_len, 4),
            "majority_class": round(float(chance), 4), "n_test": int(len(yte)),
            "mean_token_len_by_index": {int(p): round(float(lte[yte == p].mean()), 1)
                                        for p in np.unique(yte)}}
        print(f"  K={K}: hidden {acc:.4f} | token-length-only {acc_len:.4f} | majority {chance:.4f}")
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        labs = sorted(set(yte))
        ax.set_xticks(range(len(labs)), labs); ax.set_yticks(range(len(labs)), labs)
        ax.set_xlabel("predicted prefix index")
        if K == Ks[0]:
            ax.set_ylabel("true prefix index")
        ax.set_title(f"K = {K}   acc = {acc:.4f}", fontsize=8)
        ax.grid(False)
        for i in range(len(labs)):
            for j in range(len(labs)):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    fig.suptitle("Linear probe for the prefix index (layer 15, trained on train / tested on dev)",
                 y=1.03, fontsize=8)
    common.save(fig, "exp2_prefix_index_confusion.png")
    plt.close(fig)

    # ---------- (2) length control: probe within narrow token-length bins ----------
    binned = {}
    for K in Ks:
        Xtr, ytr, ltr, _ = tr[K]
        Xte, yte, lte, _ = te[K]
        edges = np.quantile(np.concatenate([ltr, lte]), [0.2, 0.4, 0.6, 0.8])
        accs = []
        for b in range(len(edges) + 1):
            lo = -np.inf if b == 0 else edges[b - 1]
            hi = np.inf if b == len(edges) else edges[b]
            mtr = (ltr > lo) & (ltr <= hi)
            mte = (lte > lo) & (lte <= hi)
            if mte.sum() < 50 or len(set(ytr[mtr])) < 2:
                continue
            a, _ = fit_probe(Xtr[mtr], ytr[mtr], Xte[mte], yte[mte])
            maj = max(np.mean(yte[mte] == c) for c in np.unique(yte[mte]))
            accs.append({"bin": b, "n_test": int(mte.sum()), "acc": round(a, 4),
                         "majority": round(float(maj), 4)})
        binned[str(K)] = accs
        print(f"  K={K} within-length-bin accs: {[a['acc'] for a in accs]}")
    res["within_length_bin_probe"] = binned

    # ---------- (3) clean probe: predict K from the question-only prefix ----------
    def first_prefix(d):
        X = np.concatenate([d[K][0][d[K][1] == 1] for K in Ks])
        y = np.concatenate([np.full((d[K][1] == 1).sum(), K) for K in Ks])
        return X, y
    Xtr, ytr = first_prefix(tr)
    Xte, yte = first_prefix(te)
    accK, cmK = fit_probe(Xtr, ytr, Xte, yte)
    majK = max(np.mean(yte == c) for c in np.unique(yte))
    res["chain_length_probe_from_question_only"] = {
        "acc": round(accK, 4), "majority_class": round(float(majK), 4),
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "confusion": cmK.tolist(), "labels": sorted(set(yte.tolist())),
        "why": "the target K is NOT present in the question-only prefix as a surface cue, "
               "unlike the prefix index, so this probe is not explained by hypothesis H2"}
    print(f"  chain-length probe from question-only prefix: {accK:.4f} (majority {majK:.4f})")

    fig, ax = plt.subplots(figsize=(common.FIG_W_CM * 0.42 * common.CM, 5.0 * common.CM))
    ax.imshow(cmK, cmap="Blues"); ax.grid(False)
    labs = sorted(set(yte.tolist()))
    ax.set_xticks(range(len(labs)), labs); ax.set_yticks(range(len(labs)), labs)
    ax.set_xlabel("predicted K"); ax.set_ylabel("true K")
    ax.set_title(f"Chain length from the question-only\nprefix   acc = {accK:.3f} "
                 f"(majority {majK:.3f})", fontsize=8)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, str(cmK[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if cmK[i, j] > cmK.max() * 0.6 else "black")
    common.save(fig, "exp2_chain_length_probe.png")
    plt.close(fig)

    out = common.Path(__file__).resolve().parent / "results" / "exp2_position_probe.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"  results -> {out}")


if __name__ == "__main__":
    main()
