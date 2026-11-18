#!/usr/bin/env bash
set -euo pipefail
setup_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
profile=${1:-}
case "$profile" in cls|seg|det) ;; *) echo 'Usage: bash uv/lock.sh {cls|seg|det}' >&2; exit 2 ;; esac
export UV_CACHE_DIR="$setup_dir/.cache"
export UV_PYTHON_INSTALL_DIR="$setup_dir/.python"
python_version=$(cat "$setup_dir/$profile/.python-version")
uv python install "$python_version" --no-bin
uv pip compile "$setup_dir/$profile/requirements.in" \
    --python "$python_version" --managed-python \
    --find-links https://download.pytorch.org/whl/torch_stable.html \
    --emit-find-links --build-constraints "$setup_dir/build-constraints.txt" \
    --custom-compile-command "bash uv/lock.sh $profile" \
    --output-file "$setup_dir/$profile/requirements.lock" --quiet
