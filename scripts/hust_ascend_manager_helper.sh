#!/usr/bin/env bash

_HUST_MANAGER_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_HUST_MANAGER_REPO_ROOT="$(cd "${_HUST_MANAGER_HELPER_DIR}/.." && pwd)"
_HUST_MANAGER_WORKSPACE_ROOT="$(cd "${_HUST_MANAGER_REPO_ROOT}/.." && pwd)"
_HUST_DEFAULT_HF_ENDPOINT="${HUST_DEFAULT_HF_ENDPOINT:-https://hf-mirror.com}"

_resolve_hust_ascend_manager_conda_python() {
  local env_prefix="${VLLM_HUST_CONDA_PREFIX:-}"
  local env_name="${VLLM_HUST_CONDA_ENV:-vllm-hust-dev}"
  local resolved_prefix

  if [[ -n "${env_prefix}" && -x "${env_prefix}/bin/python" ]]; then
    printf '%s\n' "${env_prefix}/bin/python"
    return 0
  fi

  if ! command -v conda >/dev/null 2>&1; then
    return 1
  fi

  resolved_prefix="$(conda env list 2>/dev/null | awk -v env_name="${env_name}" '$1 == env_name {print $NF; exit}')"
  if [[ -n "${resolved_prefix}" && -x "${resolved_prefix}/bin/python" ]]; then
    printf '%s\n' "${resolved_prefix}/bin/python"
    return 0
  fi

  return 1
}

_resolve_hust_ascend_manager_python() {
  if [[ -n "${VLLM_HUST_PYTHON_BIN:-}" ]]; then
    if [[ -x "${VLLM_HUST_PYTHON_BIN}" ]]; then
      printf '%s\n' "${VLLM_HUST_PYTHON_BIN}"
      return 0
    fi
    echo "[WARN] VLLM_HUST_PYTHON_BIN is set but not executable: ${VLLM_HUST_PYTHON_BIN}" >&2
  fi

  if _resolve_hust_ascend_manager_conda_python >/dev/null 2>&1; then
    _resolve_hust_ascend_manager_conda_python
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  return 1
}

hust_resolve_python_bin() {
  _resolve_hust_ascend_manager_python "$@"
}

hust_run_pip() {
  local python_bin
  python_bin="$(_resolve_hust_ascend_manager_python 2>/dev/null)" || python_bin=""

  if [[ -n "${python_bin}" ]]; then
    if ! "${python_bin}" -m pip --version >/dev/null 2>&1; then
      if ! "${python_bin}" -m ensurepip --upgrade >/dev/null 2>&1; then
        python_bin=""
      fi
    fi
  fi

  if [[ -n "${python_bin}" ]]; then
    "${python_bin}" -m pip "$@"
    return $?
  fi

  if command -v pip >/dev/null 2>&1; then
    pip "$@"
    return $?
  fi

  if command -v pip3 >/dev/null 2>&1; then
    pip3 "$@"
    return $?
  fi

  python_bin="$(_resolve_hust_ascend_manager_python 2>/dev/null)" || {
    echo "[ERROR] Could not locate python3/python for pip operations" >&2
    return 1
  }

  if ! "${python_bin}" -m pip --version >/dev/null 2>&1; then
    if ! "${python_bin}" -m ensurepip --upgrade >/dev/null 2>&1; then
      echo "[ERROR] Could not locate pip or bootstrap it with ensurepip" >&2
      return 1
    fi
  fi

  "${python_bin}" -m pip "$@"
}

_resolve_hust_ascend_manager_src() {
  local manager_repo="${HUST_ASCEND_MANAGER_REPO:-${_HUST_MANAGER_WORKSPACE_ROOT}/ascend-runtime-manager}"
  local manager_src="${manager_repo}/src"
  if [[ -d "${manager_src}" ]]; then
    printf '%s\n' "${manager_src}"
    return 0
  fi
  return 1
}

hust_apply_default_hf_mirror() {
  if [[ -n "${HF_ENDPOINT:-}" ]]; then
    return 0
  fi

  if [[ "${HUST_DISABLE_DEFAULT_HF_MIRROR:-0}" == "1" ]]; then
    return 0
  fi

  export HF_ENDPOINT="${_HUST_DEFAULT_HF_ENDPOINT}"
}

hust_ascend_manager_available() {
  if command -v hust-ascend-manager >/dev/null 2>&1; then
    return 0
  fi

  local manager_src
  local python_bin
  manager_src="$(_resolve_hust_ascend_manager_src 2>/dev/null)" || return 1
  python_bin="$(_resolve_hust_ascend_manager_python 2>/dev/null)" || return 1
  [[ -n "${manager_src}" && -n "${python_bin}" ]]
}

hust_ascend_manager_run() {
  if command -v hust-ascend-manager >/dev/null 2>&1; then
    hust-ascend-manager "$@"
    return $?
  fi

  local manager_src
  local python_bin
  manager_src="$(_resolve_hust_ascend_manager_src 2>/dev/null)" || {
    echo "[ERROR] hust-ascend-manager is required but not found in PATH" >&2
    echo "[ERROR] Could not locate a local ascend-runtime-manager workspace checkout." >&2
    return 1
  }
  python_bin="$(_resolve_hust_ascend_manager_python 2>/dev/null)" || {
    echo "[ERROR] hust-ascend-manager is required but not found in PATH" >&2
    echo "[ERROR] Could not locate a Python interpreter for the local manager fallback." >&2
    return 1
  }

  if [[ "${_HUST_ASCEND_MANAGER_FALLBACK_WARNED:-0}" != "1" ]]; then
    echo "[WARN] hust-ascend-manager not found in PATH; using local workspace fallback at ${manager_src}" >&2
    _HUST_ASCEND_MANAGER_FALLBACK_WARNED=1
  fi

  PYTHONPATH="${manager_src}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" -m hust_ascend_manager.cli "$@"
}