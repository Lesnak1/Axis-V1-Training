#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${AXIS_CLI_PYTHON:-python3}" "${ROOT}/scripts/axis.py" train "$@"
