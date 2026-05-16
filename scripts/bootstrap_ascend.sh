#!/usr/bin/env bash
set -euo pipefail

# Canonical Ascend bootstrap wrapper for the local multi-root workspace.
# Setup and launch flow lives in hust-ascend-manager.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_REF="${1:-${VLLM_HUST_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}}"
MANAGER_REPO="${HUST_ASCEND_MANAGER_REPO:-$(cd "${ASCEND_REPO_ROOT}/.." && pwd)/ascend-runtime-manager}"
MANAGER_MANIFEST="${HUST_ASCEND_MANAGER_MANIFEST:-${MANAGER_REPO}/manifests/euleros-910b.json}"
MANAGER_PYPI_SPEC="${HUST_ASCEND_MANAGER_PYPI_SPEC:-hust-ascend-manager}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/hust_ascend_manager_helper.sh"

hust_apply_default_hf_mirror

if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")

cd "${ASCEND_REPO_ROOT}"

if ! command -v hust-ascend-manager >/dev/null 2>&1 && ! hust_ascend_manager_available; then
  if [[ -f "${MANAGER_REPO}/pyproject.toml" ]]; then
    hust_run_pip install -e "${MANAGER_REPO}" --no-deps
  else
    hust_run_pip install --upgrade "${MANAGER_PYPI_SPEC}"
  fi
fi

LAUNCH_ARGS=(
  launch
  "${MODEL_REF}"
  --manifest "${MANAGER_MANIFEST}"
)

if [[ "${HUST_MANAGER_APPLY_SYSTEM:-1}" != "1" ]]; then
  LAUNCH_ARGS+=(--no-apply-system)
fi
if [[ "${HUST_MANAGER_INSTALL_PYTHON_STACK:-1}" == "1" ]]; then
  LAUNCH_ARGS+=(--install-python-stack)
fi
if [[ "${HUST_MANAGER_SKIP_SETUP:-0}" == "1" ]]; then
  LAUNCH_ARGS+=(--skip-setup)
fi
if [[ -n "${VLLM_HUST_HOST:-}" ]]; then
  LAUNCH_ARGS+=(--host "${VLLM_HUST_HOST}")
fi
if [[ -n "${VLLM_HUST_PORT:-}" ]]; then
  LAUNCH_ARGS+=(--port "${VLLM_HUST_PORT}")
fi
if [[ -n "${VLLM_HUST_SERVED_MODEL_NAME:-}" ]]; then
  LAUNCH_ARGS+=(--served-model-name "${VLLM_HUST_SERVED_MODEL_NAME}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  LAUNCH_ARGS+=(-- "${EXTRA_ARGS[@]}")
fi

hust_ascend_manager_run "${LAUNCH_ARGS[@]}"