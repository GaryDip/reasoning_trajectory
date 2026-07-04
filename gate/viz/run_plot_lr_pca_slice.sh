#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python gate/viz/plot_lr_pca_slice.py --split dev "$@"
