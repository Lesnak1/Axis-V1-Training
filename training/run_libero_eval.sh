#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-third_party/openpi}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${OPENPI_ROOT}/examples/libero/.venv/bin/python}"

if [[ ! -d "${OPENPI_ROOT}" ]]; then
  echo "OpenPI checkout not found. Set OPENPI_ROOT or install it at third_party/openpi." >&2
  exit 2
fi

if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "LIBERO Python environment not found at ${LIBERO_PYTHON}." >&2
  echo "Follow docs/LIBERO_FINETUNE_EVAL.md to initialize OpenPI submodules and sync examples/libero requirements." >&2
  exit 2
fi

cd "${OPENPI_ROOT}"
export PYTHONPATH="${OPENPI_ROOT}/third_party/libero:${PYTHONPATH:-}"
"${LIBERO_PYTHON}" examples/libero/main.py "$@"
