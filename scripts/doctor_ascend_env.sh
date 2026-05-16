#!/usr/bin/env bash
set -euo pipefail

# Canonical Ascend environment diagnosis entry.
# All diagnosis logic is centralized in hust-ascend-manager.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/hust_ascend_manager_helper.sh"

if ! hust_ascend_manager_available; then
  echo "[ERROR] hust-ascend-manager is required but not found in PATH"
  echo "[ERROR] No local ascend-runtime-manager fallback was found either."
  exit 1
fi

hust_ascend_manager_run doctor