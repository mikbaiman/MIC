#!/usr/bin/env bash
# Build the original extension into a new directory; never overwrite det/_C.
set -euo pipefail
setup_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(dirname -- "$setup_dir")
mode=${1:-cpu}
export PYTHONDONTWRITEBYTECODE=1
export MAX_JOBS=${MAX_JOBS:-2}
case "$mode" in
    cpu) export CUDA_VISIBLE_DEVICES='' ;;
    cuda)
        command -v nvcc >/dev/null || {
            echo 'CUDA build requires the CUDA 11.3 toolkit (nvcc) on PATH; the driver alone is insufficient.' >&2
            exit 1
        }
        "$setup_dir/det/.venv/bin/python" -c \
            'import torch; assert torch.cuda.is_available(), "CUDA device unavailable"'
        ;;
    *) echo 'Usage: bash uv/build-det.sh {cpu|cuda}' >&2; exit 2 ;;
esac
cd "$repo_dir"
"$setup_dir/det/.venv/bin/python" det/setup.py build_ext \
    --build-temp "$setup_dir/det/build/$mode" \
    --build-lib "$setup_dir/det/build/lib" --force
echo 'Extension built. Run: uv/det/.venv/bin/python uv/smoke.py det --device cpu'
