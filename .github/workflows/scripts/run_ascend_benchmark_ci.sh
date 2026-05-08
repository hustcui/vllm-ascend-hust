#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=${WORKSPACE_ROOT:-${GITHUB_WORKSPACE:-$PWD}}
VLLM_HUST_REPO=${VLLM_HUST_REPO:-$WORKSPACE_ROOT/vllm-hust}
VLLM_ASCEND_HUST_REPO=${VLLM_ASCEND_HUST_REPO:-$WORKSPACE_ROOT}
VLLM_HUST_BENCHMARK_REPO=${VLLM_HUST_BENCHMARK_REPO:-$WORKSPACE_ROOT/vllm-hust-benchmark}
ASCEND_HUST_TARGET_REPOSITORY=${ASCEND_HUST_TARGET_REPOSITORY:-${GITHUB_REPOSITORY:-unknown}}
ASCEND_HUST_TARGET_REF=${ASCEND_HUST_TARGET_REF:-${GITHUB_REF_NAME:-detached}}
ASCEND_HUST_TARGET_SHA=${ASCEND_HUST_TARGET_SHA:-${GITHUB_SHA:-local}}
ASCEND_HUST_TARGET_COMMIT_URL=${ASCEND_HUST_TARGET_COMMIT_URL:-${GITHUB_SERVER_URL:-https://github.com}/${ASCEND_HUST_TARGET_REPOSITORY}/commit/${ASCEND_HUST_TARGET_SHA}}
ASCEND_HUST_TARGET_SHA_SHORT=$(printf '%s' "$ASCEND_HUST_TARGET_SHA" | cut -c1-8)

RUN_ID=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-${ASCEND_HUST_TARGET_SHA_SHORT}}
RESULT_ROOT=${RESULT_ROOT:-$VLLM_ASCEND_HUST_REPO/.benchmarks/ci/$RUN_ID}
RAW_RESULT_FILE=${RAW_RESULT_FILE:-$RESULT_ROOT/raw_benchmark.json}
SUBMISSIONS_ROOT=${SUBMISSIONS_ROOT:-$RESULT_ROOT/submissions}
SUBMISSION_DIR=${SUBMISSION_DIR:-$SUBMISSIONS_ROOT/$RUN_ID}
AGGREGATE_OUTPUT_DIR=${AGGREGATE_OUTPUT_DIR:-$RESULT_ROOT/leaderboard-data}
SERVER_LOG=${SERVER_LOG:-$RESULT_ROOT/server.log}
BENCH_SCENARIO=${BENCH_SCENARIO:-random-online}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-}
BENCH_CONSTRAINTS_FILE=${BENCH_CONSTRAINTS_FILE:-}
ALLOW_RANDOM_HF_PUBLISH=${ALLOW_RANDOM_HF_PUBLISH:-0}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PARAMETERS=${MODEL_PARAMETERS:-0.5B}
MODEL_PRECISION=${MODEL_PRECISION:-BF16}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-}
DTYPE=${DTYPE:-bfloat16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-256}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-8}
BENCH_RANDOM_INPUT_LEN=${BENCH_RANDOM_INPUT_LEN:-64}
BENCH_RANDOM_OUTPUT_LEN=${BENCH_RANDOM_OUTPUT_LEN:-16}
BENCH_RANDOM_BATCH_SIZE=${BENCH_RANDOM_BATCH_SIZE:-1}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-inf}
BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-4}
BENCH_INPUT_LEN=${BENCH_INPUT_LEN:-}
BENCH_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-}
HARDWARE_VENDOR=${HARDWARE_VENDOR:-Huawei}
HARDWARE_CHIP_MODEL=${HARDWARE_CHIP_MODEL:-910B3}
CHIP_COUNT=${CHIP_COUNT:-1}
NODE_COUNT=${NODE_COUNT:-1}
PUBLISH_TO_HF=${PUBLISH_TO_HF:-0}
HF_REPO_ID=${HF_REPO_ID:-}

# shellcheck source=/dev/null
source "${VLLM_ASCEND_HUST_REPO}/scripts/hust_ascend_manager_helper.sh"

PYTHON_BIN="$(hust_resolve_python_bin 2>/dev/null)" || {
  echo "Could not locate python3/python for benchmark workflow" >&2
  exit 1
}

export PYTHONPATH="${VLLM_HUST_REPO}:${VLLM_HUST_BENCHMARK_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

server_pid=""
server_group_pid=""

cleanup() {
  if [[ -n "$server_group_pid" ]] && kill -0 "$server_group_pid" 2>/dev/null; then
    kill -TERM -- "-$server_group_pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      if ! kill -0 "$server_group_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    kill -KILL -- "-$server_group_pid" 2>/dev/null || true
  elif [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
  fi

  if [[ -n "$server_pid" ]]; then
    wait "$server_pid" || true
  fi
}

start_server() {
  if command -v setsid >/dev/null 2>&1; then
    setsid vllm serve "$MODEL_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --dtype "$DTYPE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --enforce-eager >"$SERVER_LOG" 2>&1 &
    server_pid=$!
    server_group_pid=$server_pid
  else
    vllm serve "$MODEL_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --dtype "$DTYPE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --enforce-eager >"$SERVER_LOG" 2>&1 &
    server_pid=$!
  fi
}

allocate_local_port() {
  "${PYTHON_BIN}" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

trap cleanup EXIT

if [[ -z "$PORT" ]]; then
  PORT=$(allocate_local_port)
fi

mkdir -p "$RESULT_ROOT" "$SUBMISSIONS_ROOT" "$AGGREGATE_OUTPUT_DIR"

select_idle_ascend_device() {
  "${PYTHON_BIN}" - <<'PY'
import re
import subprocess
import sys


def parse_logical_map(mapping_output: str) -> dict[tuple[str, str], int]:
  logical_map = {}
  for line in mapping_output.splitlines():
    parts = line.split()
    if len(parts) < 3:
      continue
    npu_id, chip_id, logical_id = parts[:3]
    if npu_id.isdigit() and chip_id.isdigit() and logical_id.isdigit():
      logical_map[(npu_id, chip_id)] = int(logical_id)
  return logical_map


def select_best_idle_device(mapping_output: str, info_output: str) -> int | None:
  logical_map = parse_logical_map(mapping_output)
  hbm_usage_pattern = re.compile(r"(\d+)\s*/\s*(\d+)\s*$")
  device_stats = []
  current_npu_id = None
  current_health = None

  for raw_line in info_output.splitlines():
    line = raw_line.strip()
    if not line.startswith("|"):
      continue

    parts = [part.strip() for part in line.strip("|").split("|")]
    if len(parts) < 3:
      continue

    left_column = parts[0].split()
    if len(left_column) >= 2 and left_column[0].isdigit() and parts[1] and ":" not in parts[1]:
      current_npu_id = left_column[0]
      current_health = parts[1]
      continue

    if current_npu_id is None or current_health != "OK":
      continue

    if len(left_column) != 1 or not left_column[0].isdigit() or ":" not in parts[1]:
      continue

    chip_id = left_column[0]
    logical_id = logical_map.get((current_npu_id, chip_id))
    if logical_id is None:
      continue

    hbm_match = hbm_usage_pattern.search(parts[2])
    if hbm_match is None:
      continue

    used_memory_mb = int(hbm_match.group(1))
    total_memory_mb = int(hbm_match.group(2))
    free_memory_mb = max(0, total_memory_mb - used_memory_mb)
    device_stats.append((logical_id, free_memory_mb, total_memory_mb))

  if not device_stats:
    return None

  device_stats.sort(key=lambda item: (-item[1], item[0]))
  return device_stats[0][0]


try:
  mapping_result = subprocess.run(
    ["npu-smi", "info", "-m"],
    check=False,
    capture_output=True,
    text=True,
    timeout=5,
  )
  info_result = subprocess.run(
    ["npu-smi", "info"],
    check=False,
    capture_output=True,
    text=True,
    timeout=5,
  )
except Exception:
  sys.exit(0)

if mapping_result.returncode != 0 or info_result.returncode != 0:
  sys.exit(0)

selected_device = select_best_idle_device(mapping_result.stdout, info_result.stdout)
if selected_device is not None:
  print(selected_device)
PY
}

echo "== Ascend benchmark CI =="
echo "workspace root: $WORKSPACE_ROOT"
echo "vllm-hust repo: $VLLM_HUST_REPO"
echo "vllm-ascend-hust repo: $VLLM_ASCEND_HUST_REPO"
echo "benchmark repo: $VLLM_HUST_BENCHMARK_REPO"
echo "benchmark target repository: $ASCEND_HUST_TARGET_REPOSITORY"
echo "benchmark target ref: $ASCEND_HUST_TARGET_REF"
echo "benchmark target sha: $ASCEND_HUST_TARGET_SHA"
echo "run id: $RUN_ID"
echo "result root: $RESULT_ROOT"
echo "benchmark port: $PORT"
echo "benchmark scenario: $BENCH_SCENARIO"
echo "publish to hf: $PUBLISH_TO_HF"

case "$BENCH_SCENARIO" in
  random-online)
    EFFECTIVE_INPUT_LEN=${BENCH_INPUT_LEN:-$BENCH_RANDOM_INPUT_LEN}
    EFFECTIVE_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-$BENCH_RANDOM_OUTPUT_LEN}
    EFFECTIVE_CONSTRAINTS_FILE=${BENCH_CONSTRAINTS_FILE:-$VLLM_ASCEND_HUST_REPO/.github/workflows/data/random-online-ci-constraints.json}
    bench_args=(
      --backend vllm
      --endpoint /v1/completions
      --dataset-name random
      --random-input-len "$BENCH_RANDOM_INPUT_LEN"
      --random-output-len "$BENCH_RANDOM_OUTPUT_LEN"
      --random-batch-size "$BENCH_RANDOM_BATCH_SIZE"
      --num-prompts "$BENCH_NUM_PROMPTS"
      --request-rate "$BENCH_REQUEST_RATE"
      --max-concurrency "$BENCH_MAX_CONCURRENCY"
    )
    ;;
  sharegpt-online)
    if [[ -z "$BENCH_DATASET_PATH" ]]; then
      echo "BENCH_DATASET_PATH is required for sharegpt-online" >&2
      exit 2
    fi
    if [[ -z "$BENCH_CONSTRAINTS_FILE" ]]; then
      echo "BENCH_CONSTRAINTS_FILE is required for sharegpt-online" >&2
      exit 2
    fi
    EFFECTIVE_INPUT_LEN=${BENCH_INPUT_LEN:-1024}
    EFFECTIVE_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-256}
    EFFECTIVE_CONSTRAINTS_FILE="$BENCH_CONSTRAINTS_FILE"
    bench_args=(
      --backend vllm
      --endpoint /v1/completions
      --dataset-name sharegpt
      --dataset-path "$BENCH_DATASET_PATH"
      --num-prompts "$BENCH_NUM_PROMPTS"
      --request-rate "$BENCH_REQUEST_RATE"
      --max-concurrency "$BENCH_MAX_CONCURRENCY"
    )
    ;;
  *)
    echo "Unsupported BENCH_SCENARIO: $BENCH_SCENARIO" >&2
    exit 2
    ;;
esac

if [[ "$PUBLISH_TO_HF" == "1" && "$BENCH_SCENARIO" == "random-online" && "$ALLOW_RANDOM_HF_PUBLISH" != "1" ]]; then
  echo "Refusing to publish random-online CI preview to HF without ALLOW_RANDOM_HF_PUBLISH=1" >&2
  exit 2
fi

if [[ ! -f "$EFFECTIVE_CONSTRAINTS_FILE" ]]; then
  echo "constraints file not found: $EFFECTIVE_CONSTRAINTS_FILE" >&2
  exit 2
fi

if [[ "$CHIP_COUNT" == "1" && -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  SELECTED_ASCEND_DEVICE="$(select_idle_ascend_device)"
  if [[ -n "$SELECTED_ASCEND_DEVICE" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="$SELECTED_ASCEND_DEVICE"
    export VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE="npu:0"
    echo "selected single-card Ascend device: $ASCEND_RT_VISIBLE_DEVICES"
  fi
fi

start_server

for attempt in $(seq 1 120); do
  if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null; then
    break
  fi

  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM server exited before becoming ready"
    cat "$SERVER_LOG"
    exit 1
  fi

  if [[ "$attempt" -eq 120 ]]; then
    echo "Timed out waiting for vLLM server to become ready"
    cat "$SERVER_LOG"
    exit 1
  fi

  sleep 2
done

vllm bench serve \
  --model "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  "${bench_args[@]}" \
  --save-result \
  --result-dir "$RESULT_ROOT" \
  --result-filename "$(basename "$RAW_RESULT_FILE")"

CORE_VERSION=$("${PYTHON_BIN}" - <<'PY'
import vllm
print(vllm.__version__)
PY
)

BACKEND_VERSION=$("${PYTHON_BIN}" - <<'PY'
from vllm_ascend._version import __version__
print(__version__)
PY
)

ENGINE_VERSION="$ASCEND_HUST_TARGET_SHA_SHORT"

"${PYTHON_BIN}" -m vllm_hust_benchmark.cli submit \
  "$BENCH_SCENARIO" \
  --benchmark-result-file "$RAW_RESULT_FILE" \
  --constraints-file "$EFFECTIVE_CONSTRAINTS_FILE" \
  --run-id "$RUN_ID" \
  --engine vllm-ascend-hust \
  --engine-version "$ENGINE_VERSION" \
  --model-name "$MODEL_NAME" \
  --model-parameters "$MODEL_PARAMETERS" \
  --model-precision "$MODEL_PRECISION" \
  --hardware-vendor "$HARDWARE_VENDOR" \
  --hardware-chip-model "$HARDWARE_CHIP_MODEL" \
  --chip-count "$CHIP_COUNT" \
  --node-count "$NODE_COUNT" \
  --submitter "${GITHUB_ACTOR:-ci}" \
  --data-source "vllm-ascend-hust-ci-$BENCH_SCENARIO" \
  --input-length "$EFFECTIVE_INPUT_LEN" \
  --output-length "$EFFECTIVE_OUTPUT_LEN" \
  --concurrent-requests "$BENCH_MAX_CONCURRENCY" \
  --backend-version "$BACKEND_VERSION" \
  --core-version "$CORE_VERSION" \
  --git-commit "$ASCEND_HUST_TARGET_SHA" \
  --github-commit-url "$ASCEND_HUST_TARGET_COMMIT_URL" \
  --github-repository "$ASCEND_HUST_TARGET_REPOSITORY" \
  --github-ref "$ASCEND_HUST_TARGET_REF" \
  --github-event-name "${GITHUB_EVENT_NAME:-manual}" \
  --submissions-dir "$SUBMISSIONS_ROOT"

if [[ "$PUBLISH_TO_HF" == "1" ]]; then
  if [[ -z "$HF_REPO_ID" ]]; then
    echo "HF_REPO_ID must be set when PUBLISH_TO_HF=1" >&2
    exit 2
  fi

  "${PYTHON_BIN}" -m vllm_hust_benchmark.cli sync-submission-to-hf \
    --submission-dir "$SUBMISSION_DIR" \
    --aggregate-output-dir "$AGGREGATE_OUTPUT_DIR" \
    --repo-id "$HF_REPO_ID" \
    --submissions-prefix submissions-auto \
    --commit-message "chore: sync vllm-hust benchmark from vllm-ascend-hust $RUN_ID (${ASCEND_HUST_TARGET_REPOSITORY}@${ASCEND_HUST_TARGET_REF}:${ASCEND_HUST_TARGET_SHA_SHORT})" \
    --execute
else
  "${PYTHON_BIN}" -m vllm_hust_benchmark.cli publish-website \
    --source-dir "$SUBMISSIONS_ROOT" \
    --output-dir "$AGGREGATE_OUTPUT_DIR" \
    --execute
fi

echo "RUN_ID=$RUN_ID"
echo "RAW_RESULT_FILE=$RAW_RESULT_FILE"
echo "SUBMISSION_DIR=$SUBMISSION_DIR"
echo "AGGREGATE_OUTPUT_DIR=$AGGREGATE_OUTPUT_DIR"
echo "SERVER_LOG=$SERVER_LOG"
echo "BENCH_SCENARIO=$BENCH_SCENARIO"
echo "EFFECTIVE_CONSTRAINTS_FILE=$EFFECTIVE_CONSTRAINTS_FILE"