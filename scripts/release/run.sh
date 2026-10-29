#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$ROOT:$ROOT/VMamba${PYTHONPATH:+:$PYTHONPATH}"
if [[ $# -eq 0 ]]; then
    echo 'Usage: bash scripts/release/run.sh {train|eval} [train.py/test.py arguments]' >&2
    exit 2
fi
MODE=$1
shift
cd "$ROOT/tools"
case "$MODE" in
    train) exec python train.py --cfg_file ../configs/release/vmamba_kitti.yaml "$@" ;;
    eval) exec python test.py --cfg_file ../configs/release/vmamba_kitti.yaml "$@" ;;
    *) echo "Unknown mode: $MODE (expected train or eval)" >&2; exit 2 ;;
esac
