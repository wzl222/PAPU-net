#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE="/mnt/disk1new/wzl/env/bioir/bin/python"

cd "$ROOT_DIR"
mkdir -p logs

exec "$PYTHON_EXE" run_uiqm_search.py "$@"
