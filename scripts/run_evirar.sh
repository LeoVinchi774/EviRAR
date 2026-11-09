#!/usr/bin/env sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ -x "$repo_dir/.venv/bin/evirar" ]; then
    cli="$repo_dir/.venv/bin/evirar"
else
    cli=$(command -v evirar || true)
fi
if [ -z "$cli" ] || [ ! -x "$cli" ]; then
    echo "EviRAR is not installed. Activate your environment and run: python -m pip install -e ." >&2
    exit 1
fi
exec "$cli" --config "$repo_dir/configs/default.yaml" "$@"
