#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PORT="${VECTORRF_PORT:-8765}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -c 'import vectorrf' >/dev/null 2>&1 || python -m pip install -e .
echo "VectorRF Lab: http://127.0.0.1:${PORT}"
exec python -m uvicorn vectorrf.app:app --app-dir backend --host 127.0.0.1 --port "$PORT"
