#!/bin/bash
set -euo pipefail

subcommand=${1:-}
if [[ -z "$subcommand" ]]; then
  echo "Usage: $0 <runtime-ready|same-spec|serve> [args...]" >&2
  exit 2
fi
shift || true

case "$subcommand" in
  runtime-ready)
    "${PYTHON_BIN:?PYTHON_BIN must be set}" - <<'PY'
import sys

try:
    import torch_npu
    torch_npu.npu.get_soc_version()
except Exception as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)
PY
    ;;
  same-spec)
    same_spec_runner=${1:?same-spec runner path is required}
    same_spec_spec_file=${2:?same-spec spec file path is required}
    bash "$same_spec_runner" "$same_spec_spec_file"
    ;;
  serve)
    exec env VLLM_ASCEND_TORCH_PREFLIGHT=0 \
      "${PYTHON_BIN:?PYTHON_BIN must be set}" -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_NAME:?MODEL_NAME must be set}" \
      --host "${HOST:?HOST must be set}" \
      --port "${PORT:?PORT must be set}" \
      --dtype "${DTYPE:?DTYPE must be set}" \
      --max-model-len "${MAX_MODEL_LEN:?MAX_MODEL_LEN must be set}" \
      --max-num-seqs "${MAX_NUM_SEQS:?MAX_NUM_SEQS must be set}" \
      --enforce-eager
    ;;
  *)
    echo "Unsupported subcommand: $subcommand" >&2
    exit 2
    ;;
esac