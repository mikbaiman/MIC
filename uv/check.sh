#!/usr/bin/env bash
# Run dependency checks, regressions, and model checks with the pinned environments.
set -euo pipefail
setup_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$(dirname -- "$setup_dir")"
device=cpu
integration=false
for option in "$@"; do
    case "$option" in
        --cuda) device=cuda ;;
        --integration) integration=true ;;
        *) echo 'Usage: bash uv/check.sh [--cuda] [--integration]' >&2; exit 2 ;;
    esac
done
export PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$setup_dir/.cache"
export MPLCONFIGDIR="$setup_dir/.cache/matplotlib"
export MPLBACKEND="${MPLBACKEND:-Agg}"
for profile in cls seg det; do
    uv pip check --python "$setup_dir/$profile/.venv/bin/python"
done
"$setup_dir/cls/.venv/bin/python" -m unittest discover -s uv/tests -p test_cls.py -v
"$setup_dir/det/.venv/bin/python" -m unittest discover -s uv/tests -p test_det.py -v
"$setup_dir/det/.venv/bin/python" "$setup_dir/run.py" det tests/test_metric_logger.py
"$setup_dir/det/.venv/bin/python" "$setup_dir/run.py" det tests/test_data_samplers.py
for profile in cls seg; do
    "$setup_dir/$profile/.venv/bin/python" "$setup_dir/smoke.py" "$profile" --device "$device"
done
# Detection's isolated extension is currently built for CPU.
"$setup_dir/det/.venv/bin/python" "$setup_dir/smoke.py" det --device cpu
if "$integration"; then
    "$setup_dir/cls/.venv/bin/python" "$setup_dir/integration.py" cls --device "$device"
    if [[ "$device" == cuda ]]; then
        "$setup_dir/seg/.venv/bin/python" "$setup_dir/integration.py" seg --device cuda
    else
        echo 'SKIP: segmentation CLI integration requires --cuda; its CPU model smoke test passed above.'
    fi
fi
echo 'PASS: all selected checks completed.'
