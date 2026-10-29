#!/usr/bin/env bash
# Run in a dedicated Python 3.10 environment with CUDA 11.8 nvcc available.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 is required"'
command -v nvcc >/dev/null || { echo 'CUDA 11.8 development toolkit (nvcc) is required.' >&2; exit 1; }
nvcc --version | grep -q 'release 11.8' || { echo 'Expected CUDA toolkit 11.8.' >&2; exit 1; }
export MAX_JOBS=${MAX_JOBS:-2}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
python -m pip install 'pip==24.3.1' 'setuptools==69.5.1' 'wheel==0.45.1'
python -m pip install 'torch==2.1.0' 'torchvision==0.16.0' --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements/release-cu118.txt
python -m pip install --only-binary=:all: --no-deps 'mmcv==2.1.0' \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
# mmcv's small Python dependencies; avoid mmcv-lite, which lacks CUDA ops.
python -m pip install 'addict==2.4.0' 'yapf==0.43.0'
python -m pip install --only-binary=:all: --no-deps 'torch-scatter==2.1.2+pt21cu118' \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
python -m pip install --no-build-isolation --no-deps ./VMamba/kernels/selective_scan
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
echo 'Installation complete. Run: python scripts/release/check_environment.py --cuda'
