"""Load LR gate artifacts and score delta vectors."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def load_lr_artifacts(artifacts_dir: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], dict]:
    import joblib

    artifacts_dir = artifacts_dir.resolve()
    meta_path = artifacts_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}; run fit_lr_gate.py first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    models: dict[tuple[int, int], dict[str, Any]] = {}
    for row in meta.get("models", []):
        K, j = int(row["K"]), int(row["j"])
        path = artifacts_dir / row["file"]
        models[(K, j)] = joblib.load(path)
    return models, meta


def load_pooled_artifacts(artifacts_dir: Path) -> tuple[dict[int, dict[str, Any]], dict]:
    import joblib

    artifacts_dir = artifacts_dir.resolve()
    meta_path = artifacts_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}; run fit_lr_gate_pooled.py first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    models: dict[int, dict[str, Any]] = {}
    for row in meta.get("models", []):
        j = int(row["j"])
        models[j] = joblib.load(artifacts_dir / row["file"])
    if not models:
        for f in sorted(artifacts_dir.glob("j*.joblib")):
            m = re.match(r"j(\d+)\.joblib", f.name)
            if m:
                models[int(m.group(1))] = joblib.load(f)
    return models, meta


def load_artifacts(artifacts_dir: Path, mode: str = "pooled") -> dict:
    """Retrieval-compatible loader."""
    if mode == "pooled":
        models, _ = load_pooled_artifacts(artifacts_dir)
        return models
    models, _ = load_lr_artifacts(artifacts_dir)
    return models


def score_lr_delta(delta: np.ndarray, artifact: dict[str, Any]) -> dict[str, float | bool]:
    pca = artifact["pca"]
    lr = artifact["lr"]
    tau = float(artifact["threshold"])
    Z = pca.transform(np.asarray(delta, dtype=np.float64).reshape(1, -1))
    score = float(lr.predict_proba(Z)[0, 1])
    return {"score": score, "threshold": tau, "triggered": bool(score > tau)}


def score_pooled_delta(delta: np.ndarray, artifact: dict[str, Any]) -> dict[str, float | bool]:
    return score_lr_delta(delta, artifact)


def get_gate_artifact(artifacts: dict, mode: str, K: int, j_art: int) -> dict | None:
    if mode == "pooled":
        return artifacts.get(j_art)
    return artifacts.get((K, j_art))
