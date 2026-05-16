#!/usr/bin/env bash
set -euo pipefail

# Minimal and reproducible Ascend latency benchmark entry.
# This script avoids mixed toolkit runtime by sourcing
# scripts/use_single_ascend_env.sh first.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${ASCEND_REPO_ROOT}/.." && pwd)"
VLLM_HUST_REPO="${VLLM_HUST_REPO:-${WORKSPACE_ROOT}/vllm-hust}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/hust_ascend_manager_helper.sh"

PYTHON_BIN="$(hust_resolve_python_bin 2>/dev/null)" || {
  echo "[ERROR] Could not locate python3/python for latency benchmark" >&2
  exit 1
}

hust_apply_default_hf_mirror

ASCEND_ROOT="${1:-${ASCEND_ROOT:-}}"
if [[ -n "${ASCEND_ROOT}" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/use_single_ascend_env.sh" "${ASCEND_ROOT}"
else
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/use_single_ascend_env.sh"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"

if [[ -d "${VLLM_HUST_REPO}/vllm" ]]; then
  export PYTHONPATH="${VLLM_HUST_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
fi

hust_ascend_manager_run runtime check \
  --repo "${VLLM_HUST_REPO}" \
  --require-plugin \
  --require-npu

cd "${ASCEND_REPO_ROOT}"

"${PYTHON_BIN}" - <<'PY'
import argparse
import json
import tempfile
from pathlib import Path

from vllm.benchmarks.latency import add_cli_args, main

dummy_gpt2_config = {
  "architectures": ["GPT2LMHeadModel"],
  "model_type": "gpt2",
  "vocab_size": 50257,
  "n_positions": 1024,
  "n_ctx": 1024,
  "n_embd": 64,
  "n_layer": 2,
  "n_head": 2,
  "bos_token_id": 50256,
  "eos_token_id": 50256,
  "activation_function": "gelu_new",
  "layer_norm_epsilon": 1e-5,
  "initializer_range": 0.02,
  "resid_pdrop": 0.0,
  "embd_pdrop": 0.0,
  "attn_pdrop": 0.0,
  "use_cache": True,
}

with tempfile.TemporaryDirectory(prefix="vllm-ascend-bench-") as model_dir:
  config_path = Path(model_dir) / "config.json"
  config_path.write_text(json.dumps(dummy_gpt2_config), encoding="utf-8")

  parser = argparse.ArgumentParser()
  add_cli_args(parser)

  args = parser.parse_args([
    "--model", model_dir,
    "--hf-config-path", model_dir,
    "--input-len", "32",
    "--output-len", "32",
    "--batch-size", "1",
    "--num-iters-warmup", "1",
    "--num-iters", "3",
    "--dtype", "float16",
    "--gpu-memory-utilization", "0.1",
    "--load-format", "dummy",
    "--enforce-eager",
    "--skip-tokenizer-init",
    "--disable-detokenize",
    "--compilation-config", '{"cudagraph_mode":"NONE"}',
  ])

  main(args)
PY