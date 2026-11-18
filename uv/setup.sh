#!/usr/bin/env bash
# Additive environments: never install into the repository's existing .venv.
set -euo pipefail

setup_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
profile=${1:-}
case "$profile" in
    cls|seg|det) ;;
    *) echo "Usage: bash uv/setup.sh {cls|seg|det}" >&2; exit 2 ;;
esac
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
export UV_CACHE_DIR="$setup_dir/.cache"
export UV_PYTHON_INSTALL_DIR="$setup_dir/.python"
export PYTHONDONTWRITEBYTECODE=1
python_version=$(cat "$setup_dir/$profile/.python-version")
uv python install "$python_version" --no-bin
if [[ ! -f "$setup_dir/$profile/.venv/pyvenv.cfg" ]]; then
    uv venv --python "$python_version" --managed-python "$setup_dir/$profile/.venv"
fi
uv pip sync --python "$setup_dir/$profile/.venv/bin/python" \
    --find-links https://download.pytorch.org/whl/torch_stable.html \
    --build-constraints "$setup_dir/build-constraints.txt" \
    "$setup_dir/$profile/requirements.lock"
uv pip check --python "$setup_dir/$profile/.venv/bin/python"
echo "Environment ready: $setup_dir/$profile/.venv"
echo "Smoke test: $setup_dir/$profile/.venv/bin/python $setup_dir/smoke.py $profile --device cpu"
if [[ "$profile" == det ]]; then
    echo 'Detection additionally requires its native extension; see uv/README.md.'
fi
