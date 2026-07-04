#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python gate/viz/plot_transition_3d.py --split dev --save-html "$@"
