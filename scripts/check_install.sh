#!/usr/bin/env sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ -x "$repo_dir/.venv/bin/python" ]; then
    python_bin="$repo_dir/.venv/bin/python"
    cli="$repo_dir/.venv/bin/evirar"
else
    python_bin=python
    cli=$(command -v evirar || true)
fi
if [ -z "$cli" ] || [ ! -x "$cli" ]; then
    echo "EviRAR is not installed. Activate your environment and run: python -m pip install -e '.[dev]'" >&2
    exit 1
fi
"$python_bin" -c 'import openai, yaml, dotenv; print("Python dependencies: OK")'
"$cli" --config "$repo_dir/configs/default.yaml" --check
