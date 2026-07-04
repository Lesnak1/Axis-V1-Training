#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${1:-pi05_axis_clean_local_droid_lora}"
OPENPI_ROOT="${OPENPI_ROOT:-third_party/openpi}"

if [[ ! -d "${OPENPI_ROOT}" ]]; then
  echo "OpenPI checkout not found. Set OPENPI_ROOT or install it at third_party/openpi." >&2
  exit 2
fi

if [[ "${CONFIG_NAME}" == *axis* && -z "${AXIS_CLEAN_ROOT:-}" ]]; then
  echo "AXIS_CLEAN_ROOT is required for AXIS LeRobot configs." >&2
  echo "Example: export AXIS_CLEAN_ROOT=/path/to/axis/lerobot/root" >&2
  exit 2
fi

cd "${OPENPI_ROOT}"
if [[ -n "${AXIS_CLEAN_ROOT:-}" ]]; then
  export AXIS_CLEAN_ROOT
fi

if [[ -n "${OPENPI_PYTHON:-}" ]]; then
  NORM_CMD=("${OPENPI_PYTHON}" scripts/compute_norm_stats.py)
elif [[ -x ".venv/bin/python" ]]; then
  NORM_CMD=(".venv/bin/python" scripts/compute_norm_stats.py)
elif command -v uv >/dev/null 2>&1; then
  NORM_CMD=(uv run scripts/compute_norm_stats.py)
elif command -v python >/dev/null 2>&1; then
  NORM_CMD=(python scripts/compute_norm_stats.py)
else
  echo "Could not find OpenPI Python. Set OPENPI_PYTHON or install uv/python in the OpenPI checkout." >&2
  exit 2
fi

"${NORM_CMD[@]}" --config-name "${CONFIG_NAME}" "${@:2}"
