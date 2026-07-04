#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${1:-pi05_axis_clean_local_droid_lora}"
EXP_NAME="${2:-axis_pi05_v001}"
OPENPI_ROOT="${OPENPI_ROOT:-third_party/openpi}"

if [[ ! -d "${OPENPI_ROOT}" ]]; then
  echo "OpenPI is not installed at ${OPENPI_ROOT}." >&2
  echo "Clone OpenPI to third_party/openpi or set OPENPI_ROOT, then apply openpi_patches/axis_train_openpi.patch." >&2
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
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export OPENPI_TRAIN_MODE="${OPENPI_TRAIN_MODE:-overwrite}"

if [[ -n "${OPENPI_PYTHON:-}" ]]; then
  TRAIN_CMD=("${OPENPI_PYTHON}" scripts/train.py)
elif [[ -x ".venv/bin/python" ]]; then
  TRAIN_CMD=(".venv/bin/python" scripts/train.py)
elif command -v uv >/dev/null 2>&1; then
  TRAIN_CMD=(uv run scripts/train.py)
elif command -v python >/dev/null 2>&1; then
  TRAIN_CMD=(python scripts/train.py)
else
  echo "Could not find OpenPI Python. Set OPENPI_PYTHON or install uv/python in the OpenPI checkout." >&2
  exit 2
fi

case "${OPENPI_TRAIN_MODE}" in
  overwrite)
    MODE_ARGS=(--overwrite)
    ;;
  resume)
    MODE_ARGS=(--resume)
    ;;
  none)
    MODE_ARGS=()
    ;;
  *)
    echo "OPENPI_TRAIN_MODE must be overwrite, resume, or none; got ${OPENPI_TRAIN_MODE}" >&2
    exit 2
    ;;
esac

"${TRAIN_CMD[@]}" "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  "${MODE_ARGS[@]}" \
  "${@:3}"
