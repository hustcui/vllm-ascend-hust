#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
SOURCE_HELPER=${SOURCE_HELPER:-$REPO_ROOT/.github/workflows/scripts/run_ascend_benchmark_root_helper.sh}
SYSTEM_HELPER_PATH=${SYSTEM_HELPER_PATH:-/usr/local/bin/run_ascend_benchmark_root_helper.sh}
RUNNER_USER=${RUNNER_USER:-grunner}
SUDOERS_FILE=${SUDOERS_FILE:-/etc/sudoers.d/${RUNNER_USER}-ascend-benchmark-root-helper}

require_command() {
  local cmd=${1:?command name is required}
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found: $cmd" >&2
    exit 1
  fi
}

require_command sudo
require_command visudo

if [[ ! -f "$SOURCE_HELPER" ]]; then
  echo "Benchmark root helper source not found: $SOURCE_HELPER" >&2
  exit 1
fi

tmp_helper=$(mktemp)
tmp_sudoers=$(mktemp)
cleanup() {
  rm -f "$tmp_helper" "$tmp_sudoers"
}
trap cleanup EXIT

cp "$SOURCE_HELPER" "$tmp_helper"
chmod 755 "$tmp_helper"

cat >"$tmp_sudoers" <<EOF
Defaults:${RUNNER_USER} !requiretty
${RUNNER_USER} ALL=(root) NOPASSWD:SETENV: ${SYSTEM_HELPER_PATH} *
EOF
chmod 440 "$tmp_sudoers"

sudo install -o root -g root -m 0755 "$tmp_helper" "$SYSTEM_HELPER_PATH"
sudo install -o root -g root -m 0440 "$tmp_sudoers" "$SUDOERS_FILE"

sudo visudo -cf "$SUDOERS_FILE"
sudo visudo -cf /etc/sudoers

cat <<EOF
Installed Ascend benchmark root helper.

Helper path:
  $SYSTEM_HELPER_PATH

Sudoers drop-in:
  $SUDOERS_FILE

Recommended verification (from the vllm-ascend-hust checkout):
  cd $REPO_ROOT
  sudo -u ${RUNNER_USER} -- bash -lc '\
    cd "$REPO_ROOT" && \
    source scripts/use_single_ascend_env.sh && \
    export PYTHON_BIN="\$(hust_resolve_python_bin)" && \
    sudo --preserve-env=ASCEND_AICPU_PATH,ASCEND_HOME_PATH,ASCEND_OPP_PATH,ASCEND_RT_VISIBLE_DEVICES,ASCEND_TOOLKIT_HOME,ASCEND_TOOLKIT_LATEST_HOME,ASCEND_VISIBLE_DEVICES,ATB_HOME_PATH,HCCL_CONNECT_TIMEOUT,HCCL_EXEC_TIMEOUT,LD_LIBRARY_PATH,PATH,PYTHON_BIN,PYTHONPATH -E -n \
      ${SYSTEM_HELPER_PATH} runtime-ready'

Optional GitHub Actions variable:
  VLLM_ASCEND_HUST_BENCHMARK_ROOT_HELPER=${SYSTEM_HELPER_PATH}
EOF