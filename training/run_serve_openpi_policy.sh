#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${1:?Usage: bash training/run_serve_openpi_policy.sh CONFIG_NAME CHECKPOINT_DIR}"
CHECKPOINT_DIR="${2:?Usage: bash training/run_serve_openpi_policy.sh CONFIG_NAME CHECKPOINT_DIR}"
OPENPI_ROOT="${OPENPI_ROOT:-third_party/openpi}"
OPENPI_SERVE_ENV="${OPENPI_SERVE_ENV:-}"

if [[ ! -d "${OPENPI_ROOT}" ]]; then
  echo "OpenPI checkout not found. Set OPENPI_ROOT or install it at third_party/openpi." >&2
  exit 2
fi

cd "${OPENPI_ROOT}"

if [[ -n "${OPENPI_PYTHON:-}" ]]; then
  SERVE_CMD=("${OPENPI_PYTHON}" scripts/serve_policy.py)
elif [[ -x ".venv/bin/python" ]]; then
  SERVE_CMD=(".venv/bin/python" scripts/serve_policy.py)
elif command -v uv >/dev/null 2>&1; then
  SERVE_CMD=(uv run scripts/serve_policy.py)
elif command -v python >/dev/null 2>&1; then
  SERVE_CMD=(python scripts/serve_policy.py)
else
  echo "Could not find OpenPI Python. Set OPENPI_PYTHON or install uv/python in the OpenPI checkout." >&2
  exit 2
fi

ENV_ARGS=()
if [[ -n "${OPENPI_SERVE_ENV}" ]]; then
  ENV_ARGS=(--env "${OPENPI_SERVE_ENV}")
fi

"${SERVE_CMD[@]}" "${ENV_ARGS[@]}" policy:checkpoint \
  --policy.config="${CONFIG_NAME}" \
  --policy.dir="${CHECKPOINT_DIR}" \
  "${@:3}"
