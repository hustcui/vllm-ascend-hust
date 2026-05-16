#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source scripts/use_single_ascend_env.sh [ASCEND_TOOLKIT_ROOT]
#
# Runtime resolution and exports are centralized in hust-ascend-manager.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/hust_ascend_manager_helper.sh"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[ERROR] This script must be sourced, not executed."
  echo "[ERROR] Use: source scripts/use_single_ascend_env.sh [ASCEND_TOOLKIT_ROOT]"
  exit 1
fi

if ! hust_ascend_manager_available; then
  echo "[ERROR] hust-ascend-manager is required but not found in PATH"
  echo "[ERROR] No local ascend-runtime-manager fallback was found either."
  return 1
fi

ASCEND_ROOT_ARG="${1:-}"
if [[ -n "${ASCEND_ROOT_ARG}" ]]; then
  eval "$(hust_ascend_manager_run env --shell --ascend-root "${ASCEND_ROOT_ARG}")"
else
  eval "$(hust_ascend_manager_run env --shell)"
fi

if [[ -n "${HUST_ATB_SET_ENV:-}" && -f "${HUST_ATB_SET_ENV}" ]]; then
  set +u
  source "${HUST_ATB_SET_ENV}" --cxx_abi=1
  set -u
fi

if [[ "${HUST_ASCEND_HAS_STREAM_ATTR:-0}" != "1" ]]; then
  echo "[WARN] Current Ascend runtime does not export aclrtSetStreamAttribute"
  echo "[WARN] npugraph_ex requires a newer CANN runtime. vllm-ascend currently recommends CANN 8.5.1."
fi

if [[ -z "${HUST_ASCEND_RUNTIME_VERSION:-}" ]]; then
  echo "[WARN] Could not detect the exact CANN runtime version from hust-ascend-manager output."
  echo "[WARN] Current ASCEND_HOME_PATH resolves to ${ASCEND_HOME_PATH:-<unset>}."
fi

if [[ "${HUST_REQUIRE_NPUGRAPH:-0}" == "1" && "${HUST_ASCEND_HAS_STREAM_ATTR:-0}" != "1" ]]; then
  echo "[ERROR] HUST_REQUIRE_NPUGRAPH=1 but current runtime cannot support npugraph_ex"
  return 1
fi

echo "[OK] Single Ascend runtime is configured"
echo "  ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}"
echo "  CANN_VERSION=${HUST_ASCEND_RUNTIME_VERSION:-<unknown>}"
echo "  HAS_aclrtSetStreamAttribute=${HUST_ASCEND_HAS_STREAM_ATTR:-0}"