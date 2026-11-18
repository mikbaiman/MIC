#!/usr/bin/env bash
set -euo pipefail
setup_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
profile=${1:-}
case "$profile" in cls|seg|det) ;; *) echo 'Usage: bash uv/run.sh {cls|seg|det} script.py [arguments...]' >&2; exit 2 ;; esac
if [[ ! -x "$setup_dir/$profile/.venv/bin/python" ]]; then
    echo "Run bash uv/setup.sh $profile first." >&2
    exit 1
fi
export UV_CACHE_DIR="$setup_dir/.cache"
export UV_PYTHON_INSTALL_DIR="$setup_dir/.python"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$setup_dir/.cache/matplotlib"
export PATH="$setup_dir/$profile/.venv/bin:$PATH"
exec uv run --no-project --python "$setup_dir/$profile/.venv/bin/python" \
    "$setup_dir/run.py" "$@"
