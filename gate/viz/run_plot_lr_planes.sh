#!/usr/bin/env bash
# Generate LR decision-plane figures for pooled gate (all j).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python gate/viz/plot_lr_planes.py --split dev "$@"
