"""
Global corpus embedding cache: precompute + persist embeddings for EVERY
passage across a full dataset split, once, so a GlobalPoolSource
(candidate_pool.py) never re-encodes a passage per query. Built by the
build_embedding_cache.py CLI (a separate, explicit step — NOT built lazily
inside an experiment run, since a first build can take a long time and
shouldn't be a surprise buried in a run_retrieval_exp_wavefront_v2.py
invocation).

Two storage shapes, because cosine and ColBERT represent a passage very
differently:

- cosine: one dense vector per passage -> a single (N, dim) float32 matrix.
  At the corpus scale this repo's datasets top out at (musique/2wiki/hotpot
  dev, upper bound ~50k-125k passages, BGE-base 768-dim -> well under 1GB),
  brute-force matmul is comfortably sufficient; no FAISS/ANN needed.
- ColBERT: a variable number of token-level vectors per passage (typically
  ~128-dim, up to ~180 tokens/passage for colbert-ir/colbertv2.0's defaults —
  confirm against whatever colbert-ai version is actually installed, this
  wasn't available to test against in the environment this was written in).
  Padding every passage to the corpus's longest passage would waste a lot of
  space, so this is stored ragged: one big (total_tokens, dim) matrix plus an
  (N+1,) offsets array, passage i's vectors are embeddings[offsets[i]:offsets[i+1]].
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_retrieval_exp import load_2wiki, load_hotpot, load_musique  # noqa: E402


def sanitize(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name).strip("_") or "x"


def cache_dir_for(
    cache_root: Path, *, dataset: str, split: str, retriever_kind: str, model_name: str
) -> Path:
    return cache_root / f"{dataset}_{split}_{retriever_kind}_{sanitize(model_name)}"


def _load_records(
    dataset: str,
    split: str,
    *,
    musique_dir: Path,
    twowiki_file: Path,
    hotpot_file: Path,
) -> dict[str, dict]:
    if dataset == "musique":
        return load_musique(musique_dir, split)
    if dataset == "2wiki":
        return load_2wiki(twowiki_file)
    if dataset == "hotpot":
        return load_hotpot(hotpot_file)
    raise ValueError(f"unknown dataset: {dataset}")


def _source_file_for(
    dataset: str, split: str, *, musique_dir: Path, twowiki_file: Path, hotpot_file: Path
) -> Path:
    if dataset == "musique":
        return musique_dir / f"musique_ans_v1.0_{split}.jsonl"
    if dataset == "2wiki":
        return twowiki_file
    if dataset == "hotpot":
        return hotpot_file
    raise ValueError(f"unknown dataset: {dataset}")


def _fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime": st.st_mtime}


def collect_passages(records: dict[str, dict]) -> list[dict[str, Any]]:
    """Union every record's paragraphs into one deterministic-order list,
    each tagged with the "_source_id" of the example it actually belongs to.
    No de-duplication across examples (matches the corpus-scale estimate this
    was sized against — see 0705update-style sizing notes in the design
    plan); revisit if the resulting matrix turns out too large in practice."""
    passages: list[dict[str, Any]] = []
    for source_id in sorted(records):
        row = records[source_id]
        for p in row.get("paragraphs") or []:
            passages.append({
                "_source_id": source_id,
                "idx": int(p.get("idx", -1)),
                "title": p.get("title") or "",
                "paragraph_text": p.get("paragraph_text") or "",
            })
    return passages


def _passage_doc_text(p: dict[str, Any]) -> str:
    """Same template as run_retrieval_exp.embed_retrieval, so cached cosine
    vectors are numerically identical to what encoding the same passage
    on-the-fly (local-pool path) would produce."""
    return f"{p.get('title') or ''}\n{(p.get('paragraph_text') or '').strip()}"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_cosine_cache(
    *,
    dataset: str,
    split: str,
    model_name: str,
    cache_root: Path,
    musique_dir: Path,
    twowiki_file: Path,
    hotpot_file: Path,
    encode_batch_size: int = 256,
) -> Path:
    from sentence_transformers import SentenceTransformer

    records = _load_records(
        dataset, split, musique_dir=musique_dir, twowiki_file=twowiki_file, hotpot_file=hotpot_file,
    )
    passages = collect_passages(records)
    print(f"[embedding_cache] {dataset}/{split}: {len(records)} examples, {len(passages)} passages")

    st = SentenceTransformer(model_name)
    docs = [_passage_doc_text(p) for p in passages]
    embeddings = st.encode(
        docs, batch_size=encode_batch_size, normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)

    out_dir = cache_dir_for(
        cache_root, dataset=dataset, split=split, retriever_kind="cosine", model_name=model_name,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    with (out_dir / "passages.jsonl").open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    src = _source_file_for(
        dataset, split, musique_dir=musique_dir, twowiki_file=twowiki_file, hotpot_file=hotpot_file,
    )
    manifest = {
        "dataset": dataset, "split": split, "retriever_kind": "cosine", "model_name": model_name,
        "dim": int(embeddings.shape[1]), "n_passages": len(passages),
        "source_fingerprint": [_fingerprint(src)],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[embedding_cache] wrote {out_dir} ({embeddings.nbytes / 1e6:.1f}MB embeddings)")
    return out_dir


def build_colbert_cache(
    *,
    dataset: str,
    split: str,
    model_name: str,
    cache_root: Path,
    musique_dir: Path,
    twowiki_file: Path,
    hotpot_file: Path,
    encode_batch_size: int = 32,
) -> Path:
    import torch
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint

    records = _load_records(
        dataset, split, musique_dir=musique_dir, twowiki_file=twowiki_file, hotpot_file=hotpot_file,
    )
    passages = collect_passages(records)
    print(f"[embedding_cache] {dataset}/{split}: {len(records)} examples, {len(passages)} passages")

    config = ColBERTConfig()
    ckpt = Checkpoint(model_name, colbert_config=config)

    dim: int | None = None
    chunks: list[np.ndarray] = []
    lengths: list[int] = []
    docs = [_passage_doc_text(p) for p in passages]

    for start in range(0, len(docs), encode_batch_size):
        batch_docs = docs[start : start + encode_batch_size]
        D = ckpt.docFromText(batch_docs, keep_dims=True)
        if isinstance(D, tuple):
            D, mask = D
        else:
            mask = None
        if dim is None:
            dim = int(D.shape[-1])
        # keep_dims=True pads every doc in the batch to that batch's own max
        # length; recover each doc's true (non-padded) token count from the
        # attention mask so the on-disk cache is ragged, not padded to the
        # whole corpus's longest passage.
        if mask is not None:
            true_lens = mask.sum(dim=1).tolist()
        else:
            true_lens = [D.shape[1]] * D.shape[0]
        for i, true_len in enumerate(true_lens):
            true_len = max(1, int(true_len))
            vec = D[i, :true_len].to(torch.float16).cpu().numpy()
            chunks.append(vec)
            lengths.append(true_len)
        if (start // encode_batch_size) % 20 == 0:
            print(f"[embedding_cache]   colbert encode {start + len(batch_docs)}/{len(docs)}", flush=True)

    token_embeddings = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, dim or 128), dtype=np.float16)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    out_dir = cache_dir_for(
        cache_root, dataset=dataset, split=split, retriever_kind="colbert", model_name=model_name,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "token_embeddings.npy", token_embeddings)
    np.save(out_dir / "offsets.npy", offsets)
    with (out_dir / "passages.jsonl").open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    src = _source_file_for(
        dataset, split, musique_dir=musique_dir, twowiki_file=twowiki_file, hotpot_file=hotpot_file,
    )
    manifest = {
        "dataset": dataset, "split": split, "retriever_kind": "colbert", "model_name": model_name,
        "dim": int(dim or 128), "n_passages": len(passages),
        "source_fingerprint": [_fingerprint(src)],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[embedding_cache] wrote {out_dir} "
        f"({token_embeddings.nbytes / 1e6:.1f}MB token embeddings, {len(passages)} passages)"
    )
    return out_dir


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

@dataclass
class GlobalEmbeddingCache:
    dataset: str
    split: str
    retriever_kind: str
    model_name: str
    paragraphs: list[dict[str, Any]]
    precomputed: Any  # np.ndarray for cosine; (token_embeddings, offsets) for colbert

    @classmethod
    def load(
        cls,
        *,
        dataset: str,
        split: str,
        retriever_kind: str,
        model_name: str,
        cache_root: Path,
        musique_dir: Path | None = None,
        twowiki_file: Path | None = None,
        hotpot_file: Path | None = None,
        check_freshness: bool = True,
    ) -> "GlobalEmbeddingCache":
        out_dir = cache_dir_for(
            cache_root, dataset=dataset, split=split, retriever_kind=retriever_kind, model_name=model_name,
        )
        manifest_path = out_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No embedding cache at {out_dir}. Build it first:\n"
                f"  python build_embedding_cache.py --dataset {dataset} --split {split} "
                f"--retriever {retriever_kind} --model {model_name}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if check_freshness and musique_dir is not None:
            src = _source_file_for(
                dataset, split, musique_dir=musique_dir,
                twowiki_file=twowiki_file, hotpot_file=hotpot_file,
            )
            current = _fingerprint(src)
            cached = (manifest.get("source_fingerprint") or [{}])[0]
            if current.get("size") != cached.get("size") or current.get("mtime") != cached.get("mtime"):
                raise RuntimeError(
                    f"Embedding cache at {out_dir} looks stale (source file {src} changed since "
                    "it was built) — rebuild with build_embedding_cache.py before using it."
                )

        passages = []
        with (out_dir / "passages.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    passages.append(json.loads(line))

        if retriever_kind == "cosine":
            precomputed = np.load(out_dir / "embeddings.npy")
        elif retriever_kind == "colbert":
            precomputed = (
                np.load(out_dir / "token_embeddings.npy"),
                np.load(out_dir / "offsets.npy"),
            )
        else:
            raise ValueError(f"unknown retriever_kind: {retriever_kind}")

        return cls(
            dataset=dataset, split=split, retriever_kind=retriever_kind, model_name=model_name,
            paragraphs=passages, precomputed=precomputed,
        )


DEFAULT_CACHE_ROOT = HERE / "embedding_cache"
