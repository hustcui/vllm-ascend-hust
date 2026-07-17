#!/bin/bash
set -euo pipefail

WORKSPACE_ROOT=${WORKSPACE_ROOT:-${GITHUB_WORKSPACE:-$PWD}}
VLLM_ASCEND_HUST_REPO=${VLLM_ASCEND_HUST_REPO:-$WORKSPACE_ROOT}
VLLM_HUST_DEV_HUB_REPO=${VLLM_HUST_DEV_HUB_REPO:-$WORKSPACE_ROOT/vllm-hust-dev-hub}
VLLM_HUST_CONDA_ENV=${VLLM_HUST_CONDA_ENV:-vllm-hust-dev}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
DEV_HUB_QUICKSTART_CONDA=${DEV_HUB_QUICKSTART_CONDA:-1}

detect_cann_major_version() {
  local candidate
  local version
  for candidate in \
    "${ASCEND_HOME_PATH:-}/version.info" \
    "${ASCEND_HOME_PATH:-}/runtime/version.info" \
    "/usr/local/Ascend/ascend-toolkit/latest/version.info" \
    "/usr/local/Ascend/ascend-toolkit/latest/runtime/version.info" \
    "${CONDA_PREFIX:-}/Ascend/cann/version.info" \
    "${CONDA_PREFIX:-}/Ascend/cann/runtime/version.info"; do
    if [[ -f "$candidate" ]]; then
      version="$(grep -oE '[0-9]+[.][0-9]+([.][0-9]+)?' "$candidate" | head -n 1 || true)"
      if [[ -n "$version" ]]; then
        printf '%s\n' "${version%%.*}"
        return 0
      fi
    fi
  done
  return 1
}

resolve_compile_custom_kernels() {
  local requested="${COMPILE_CUSTOM_KERNELS:-auto}"
  local cann_major

  if [[ "$requested" == "auto" ]]; then
    if [[ "${HUST_ASCEND_TBE_AVAILABLE:-1}" != "1" ]]; then
      printf '0\n'
      return 0
    fi
    cann_major="$(detect_cann_major_version || true)"
    if [[ "$cann_major" == "9" ]]; then
      printf 'dev-hub-default\n'
    else
      printf '0\n'
    fi
    return 0
  fi

  if [[ ! "$requested" =~ ^[0-9]+$ ]]; then
    echo "Invalid COMPILE_CUSTOM_KERNELS: $requested" >&2
    return 2
  fi

  printf '%s\n' "$requested"
}

ensure_ascend_repo_workspace_entry() {
  local workspace_entry="$WORKSPACE_ROOT/vllm-ascend-hust"
  local entry_real
  local repo_real

  repo_real="$(cd "$VLLM_ASCEND_HUST_REPO" && pwd -P)"

  if [[ -L "$workspace_entry" ]]; then
    ln -sfn "$VLLM_ASCEND_HUST_REPO" "$workspace_entry"
    return 0
  fi

  if [[ -e "$workspace_entry" ]]; then
    entry_real="$(cd "$workspace_entry" && pwd -P)"
    if [[ "$entry_real" != "$repo_real" ]]; then
      echo "Workspace entry $workspace_entry points to $entry_real, expected $repo_real" >&2
      return 2
    fi
    return 0
  fi

  ln -s "$VLLM_ASCEND_HUST_REPO" "$workspace_entry"
}

if [[ ! -f "$VLLM_HUST_DEV_HUB_REPO/scripts/quickstart.sh" ]]; then
  echo "dev-hub quickstart not found: $VLLM_HUST_DEV_HUB_REPO/scripts/quickstart.sh" >&2
  exit 2
fi

if [[ ! -f "$WORKSPACE_ROOT/ascend-runtime-manager/pyproject.toml" ]]; then
  echo "ascend-runtime-manager checkout not found under workspace: $WORKSPACE_ROOT/ascend-runtime-manager" >&2
  exit 2
fi

if [[ -f "$VLLM_ASCEND_HUST_REPO/scripts/use_single_ascend_env.sh" ]]; then
  export HUST_REQUIRE_CANN_TBE="${HUST_REQUIRE_CANN_TBE:-0}"
  # shellcheck source=/dev/null
  source "$VLLM_ASCEND_HUST_REPO/scripts/use_single_ascend_env.sh"
fi

export HUST_DEV_HUB_SKIP_ASCEND_SYSTEM_APPLY=1

ensure_ascend_repo_workspace_entry

echo "Using dev-hub quickstart: $VLLM_HUST_DEV_HUB_REPO/scripts/quickstart.sh"

quickstart_args=(
  --install
  --install-mode refresh
  --install-scope full
  --env-name "$VLLM_HUST_CONDA_ENV"
  -y
)

if [[ "$DEV_HUB_QUICKSTART_CONDA" == "1" ]]; then
  quickstart_args=(
    --conda
    "${quickstart_args[@]}"
    --python "$PYTHON_VERSION"
  )
fi

requested_compile_custom_kernels="${COMPILE_CUSTOM_KERNELS:-auto}"
resolved_compile_custom_kernels="$(resolve_compile_custom_kernels)"
case "$resolved_compile_custom_kernels" in
  dev-hub-default)
    unset HUST_DEV_HUB_ASCEND_COMPILE_CUSTOM_KERNELS
    echo "COMPILE_CUSTOM_KERNELS=auto resolved to dev-hub default policy for CANN 9"
    ;;
  0)
    export COMPILE_CUSTOM_KERNELS=0
    export HUST_DEV_HUB_ASCEND_COMPILE_CUSTOM_KERNELS=0
    quickstart_args+=(--ascend-lightweight)
    if [[ "$requested_compile_custom_kernels" == "auto" && "${HUST_ASCEND_TBE_AVAILABLE:-1}" != "1" ]]; then
      echo "COMPILE_CUSTOM_KERNELS=auto resolved to lightweight mode because TBE is unavailable on this runner"
    elif [[ "$requested_compile_custom_kernels" == "auto" ]]; then
      echo "COMPILE_CUSTOM_KERNELS=auto resolved to lightweight mode for CANN 8.x or unknown runtime"
    else
      echo "Using configured COMPILE_CUSTOM_KERNELS=0"
    fi
    ;;
  *)
    export COMPILE_CUSTOM_KERNELS="$resolved_compile_custom_kernels"
    export HUST_DEV_HUB_ASCEND_COMPILE_CUSTOM_KERNELS="$resolved_compile_custom_kernels"
    quickstart_args+=(--ascend-custom-kernels "$resolved_compile_custom_kernels")
    echo "Using configured COMPILE_CUSTOM_KERNELS=$resolved_compile_custom_kernels"
    ;;
esac

bash "$VLLM_HUST_DEV_HUB_REPO/scripts/quickstart.sh" "${quickstart_args[@]}"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "COMPILE_CUSTOM_KERNELS=${COMPILE_CUSTOM_KERNELS:-auto}" >> "$GITHUB_ENV"
fi
