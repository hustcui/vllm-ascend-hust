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
RUNTIME_READY_LOG=${RUNTIME_READY_LOG:-$RESULT_ROOT/runtime-ready.log}
CI_RUNTIME_ROOT=${CI_RUNTIME_ROOT:-$WORKSPACE_ROOT/.ci-runtime}
PROCESS_MARKER_DIR=${PROCESS_MARKER_DIR:-$CI_RUNTIME_ROOT/process-markers}
SERVER_PID_MARKER=${SERVER_PID_MARKER:-$PROCESS_MARKER_DIR/ascend-benchmark-server.pid}
SERVER_PGID_MARKER=${SERVER_PGID_MARKER:-$PROCESS_MARKER_DIR/ascend-benchmark-server.pgid}
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

CI_HOME=${CI_HOME:-$WORKSPACE_ROOT/.ci-home}
HOME=$CI_HOME
XDG_CACHE_HOME=$CI_HOME/.cache
XDG_CONFIG_HOME=$CI_HOME/.config
export CI_HOME HOME XDG_CACHE_HOME XDG_CONFIG_HOME

export PYTHONPATH="${VLLM_HUST_REPO}:${VLLM_HUST_BENCHMARK_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
VLLM_CLI=("${PYTHON_BIN}" -m vllm.entrypoints.cli.main)
VLLM_SERVE=("${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server)
SERVER_READY_TIMEOUT_SECONDS=${SERVER_READY_TIMEOUT_SECONDS:-600}
SERVER_READY_POLL_SECONDS=${SERVER_READY_POLL_SECONDS:-2}
SERVER_START_RETRIES=${SERVER_START_RETRIES:-8}
SERVER_START_RETRY_DELAY_SECONDS=${SERVER_START_RETRY_DELAY_SECONDS:-10}
ASCEND_RUNTIME_READY_TIMEOUT_SECONDS=${ASCEND_RUNTIME_READY_TIMEOUT_SECONDS:-30}
ASCEND_RUNTIME_READY_POLL_SECONDS=${ASCEND_RUNTIME_READY_POLL_SECONDS:-10}
RESOURCE_BUSY_EXIT_CODE=${RESOURCE_BUSY_EXIT_CODE:-75}

server_pid=""
server_group_pid=""

cleanup() {
  if [[ -n "$server_group_pid" ]]; then
    kill -TERM -- "-$server_group_pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      if ! kill -0 -- "-$server_group_pid" 2>/dev/null; then
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

  server_pid=""
  server_group_pid=""
  rm -f "$SERVER_PID_MARKER" "$SERVER_PGID_MARKER"
}

server_log_indicates_resource_busy() {
  [[ -f "$SERVER_LOG" ]] && grep -qE 'Resource_Busy\(EL0005\)|aclInit, error code is 507899|The resources are busy' "$SERVER_LOG"
}

runtime_ready_log_indicates_resource_busy() {
  [[ -f "$RUNTIME_READY_LOG" ]] && grep -qE 'Resource_Busy\(EL0005\)|aclInit, error code is 507899|The resources are busy' "$RUNTIME_READY_LOG"
}

cleanup_previous_ci_processes() {
  local marker_pgid marker_pid remaining_matches remaining_pids

  if [[ -f "$SERVER_PGID_MARKER" ]]; then
    marker_pgid=$(tr -d '[:space:]' <"$SERVER_PGID_MARKER")
    if [[ -n "$marker_pgid" ]] && kill -0 "$marker_pgid" 2>/dev/null; then
      echo "Cleaning leftover Ascend benchmark process group: $marker_pgid"
      kill -TERM -- "-$marker_pgid" 2>/dev/null || true
      for _ in $(seq 1 10); do
        if ! kill -0 "$marker_pgid" 2>/dev/null; then
          break
        fi
        sleep 1
      done
      kill -KILL -- "-$marker_pgid" 2>/dev/null || true
    fi
  fi

  if [[ -f "$SERVER_PID_MARKER" ]]; then
    marker_pid=$(tr -d '[:space:]' <"$SERVER_PID_MARKER")
    if [[ -n "$marker_pid" ]] && kill -0 "$marker_pid" 2>/dev/null; then
      echo "Cleaning leftover Ascend benchmark process: $marker_pid"
      kill "$marker_pid" 2>/dev/null || true
    fi
  fi

  remaining_matches=$(ps -eo pid,ppid,pgid,sid,etimes,args \
    | grep -F "$WORKSPACE_ROOT" \
    | grep -E 'vllm|python|pytest' \
    | grep -v grep || true)
  if [[ -n "$remaining_matches" ]]; then
    echo "Remaining workspace-scoped vLLM/Python processes before benchmark:"
    echo "$remaining_matches"
    remaining_pids=$(printf '%s\n' "$remaining_matches" | awk '{print $1}')
    if [[ -n "$remaining_pids" ]]; then
      echo "Cleaning workspace-scoped leftover process(es): $remaining_pids"
      # shellcheck disable=SC2086
      kill -TERM $remaining_pids 2>/dev/null || true
      for _ in $(seq 1 10); do
        if ! ps -p "$(printf '%s' "$remaining_pids" | paste -sd, -)" >/dev/null 2>&1; then
          break
        fi
        sleep 1
      done
      # shellcheck disable=SC2086
      kill -KILL $remaining_pids 2>/dev/null || true
    fi
  else
    echo "No leftover workspace-scoped vLLM/Python processes detected before benchmark."
  fi

  rm -f "$SERVER_PID_MARKER" "$SERVER_PGID_MARKER"
}

wait_for_ascend_runtime_ready() {
  local max_attempts
  max_attempts=$(((ASCEND_RUNTIME_READY_TIMEOUT_SECONDS + ASCEND_RUNTIME_READY_POLL_SECONDS - 1) / ASCEND_RUNTIME_READY_POLL_SECONDS))
  if (( max_attempts < 1 )); then
    max_attempts=1
  fi

  for runtime_attempt in $(seq 1 "$max_attempts"); do
    if "${PYTHON_BIN}" - <<'PY' >"$RUNTIME_READY_LOG" 2>&1
import sys

try:
    import torch_npu
    torch_npu.npu.get_soc_version()
except Exception as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)
PY
    then
      return 0
    fi

    cat "$RUNTIME_READY_LOG" >&2

    if [[ "$runtime_attempt" -eq "$max_attempts" ]]; then
      if runtime_ready_log_indicates_resource_busy; then
        return "$RESOURCE_BUSY_EXIT_CODE"
      fi
      return 1
    fi

    echo "Ascend runtime not ready yet; waiting ${ASCEND_RUNTIME_READY_POLL_SECONDS}s before retrying device initialization (${runtime_attempt}/${max_attempts})"
    sleep "$ASCEND_RUNTIME_READY_POLL_SECONDS"
  done
}

resolve_npu_smi_bin() {
  local candidate
  if candidate="$(command -v npu-smi 2>/dev/null)" && [[ -n "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  for candidate in /usr/local/bin/npu-smi /usr/local/sbin/npu-smi /usr/sbin/npu-smi /usr/bin/npu-smi; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

start_server() {
  if command -v setsid >/dev/null 2>&1; then
    setsid env VLLM_ASCEND_TORCH_PREFLIGHT=0 "${VLLM_SERVE[@]}" \
      --model "$MODEL_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --dtype "$DTYPE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --enforce-eager >"$SERVER_LOG" 2>&1 &
    server_pid=$!
    server_group_pid=$server_pid
    printf '%s\n' "$server_pid" >"$SERVER_PID_MARKER"
    printf '%s\n' "$server_group_pid" >"$SERVER_PGID_MARKER"
  else
    env VLLM_ASCEND_TORCH_PREFLIGHT=0 "${VLLM_SERVE[@]}" \
      --model "$MODEL_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --dtype "$DTYPE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --enforce-eager >"$SERVER_LOG" 2>&1 &
    server_pid=$!
    printf '%s\n' "$server_pid" >"$SERVER_PID_MARKER"
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

mkdir -p "$RESULT_ROOT" "$SUBMISSIONS_ROOT" "$AGGREGATE_OUTPUT_DIR" "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$PROCESS_MARKER_DIR"
cleanup_previous_ci_processes

NPU_SMI_BIN="$(resolve_npu_smi_bin 2>/dev/null || true)"
if [[ -n "$NPU_SMI_BIN" ]]; then
  echo "Using npu-smi binary: $NPU_SMI_BIN"
else
  echo "Could not resolve npu-smi binary; single-card device selection may be unavailable"
fi

select_ascend_device() {
  ASCEND_DEVICE_SELECTION_ATTEMPT="${1:-1}" NPU_SMI_BIN="${2:-}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
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


def list_logical_devices(mapping_output: str) -> list[int]:
  logical_devices = set(parse_logical_map(mapping_output).values())
  return sorted(logical_devices)


def list_status_devices(info_output: str) -> list[int]:
  status_devices = set()
  for raw_line in info_output.splitlines():
    line = raw_line.strip()
    if not line.startswith("|"):
      continue

    parts = [part.strip() for part in line.strip("|").split("|")]
    if len(parts) < 2:
      continue

    left_column = parts[0].split()
    if len(left_column) >= 2 and left_column[0].isdigit() and parts[1] and ":" not in parts[1]:
      status_devices.add(int(left_column[0]))

  return sorted(status_devices)


def list_devnode_devices() -> list[int]:
  devnode_devices = set()
  for device_path in Path("/dev").glob("davinci[0-9]*"):
    suffix = device_path.name.removeprefix("davinci")
    if suffix.isdigit():
      devnode_devices.add(int(suffix))
  return sorted(devnode_devices)


def run_npu_smi(*args: str) -> subprocess.CompletedProcess[str] | None:
  npu_smi_bin = os.environ.get("NPU_SMI_BIN")
  if not npu_smi_bin:
    return None

  # Probe npu-smi with a minimal environment so Ascend/Python job variables do
  # not interfere with the management CLI.
  clean_env = {
    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    "HOME": os.environ.get("HOME", ""),
    "LANG": os.environ.get("LANG", "C.UTF-8"),
    "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
  }

  try:
    return subprocess.run(
      [npu_smi_bin, *args],
      check=False,
      capture_output=True,
      text=True,
      timeout=5,
      env=clean_env,
    )
  except subprocess.TimeoutExpired:
    print(f"npu-smi {' '.join(args)} timed out after 5s", file=sys.stderr)
    return None
  except Exception as exc:
    print(f"npu-smi {' '.join(args)} failed: {exc}", file=sys.stderr)
    return None


def select_best_idle_device(info_output: str, logical_map: dict[tuple[str, str], int]) -> tuple[int, str] | None:
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
    device_source = "idle"
    if logical_id is None:
      if chip_id != "0":
        continue
      logical_id = int(current_npu_id)
      device_source = "status-fallback"

    hbm_match = hbm_usage_pattern.search(parts[2])
    if hbm_match is None:
      continue

    used_memory_mb = int(hbm_match.group(1))
    total_memory_mb = int(hbm_match.group(2))
    free_memory_mb = max(0, total_memory_mb - used_memory_mb)
    device_stats.append((logical_id, free_memory_mb, total_memory_mb, device_source))

  if not device_stats:
    return None

  device_stats.sort(key=lambda item: (-item[1], item[0], item[3]))
  selected_device, _, _, selected_source = device_stats[0]
  return selected_device, selected_source

mapping_result = run_npu_smi("info", "-m")
logical_map = {}
logical_devices = []
if mapping_result is not None and mapping_result.returncode == 0:
  logical_map = parse_logical_map(mapping_result.stdout)
  logical_devices = list_status_devices(mapping_result.stdout)
elif mapping_result is not None:
  print(f"npu-smi info -m returned {mapping_result.returncode}: {mapping_result.stderr.strip()}", file=sys.stderr)

selection_attempt = max(1, int(os.environ.get("ASCEND_DEVICE_SELECTION_ATTEMPT", "1")))

info_result = run_npu_smi("info")
if info_result is not None and info_result.returncode == 0:
  selected_device = select_best_idle_device(info_result.stdout, logical_map)
  if selected_device is not None:
    device_id, device_source = selected_device
    print(f"{device_id}\t{device_source}")
    sys.exit(0)
  status_devices = list_status_devices(info_result.stdout)
  if status_devices:
    fallback_device = status_devices[(selection_attempt - 1) % len(status_devices)]
    print(f"{fallback_device}\tstatus-round-robin")
    sys.exit(0)
elif info_result is not None:
  print(f"npu-smi info returned {info_result.returncode}: {info_result.stderr.strip()}", file=sys.stderr)

if logical_devices:
  fallback_device = logical_devices[(selection_attempt - 1) % len(logical_devices)]
  print(f"{fallback_device}\tfallback-round-robin")
  sys.exit(0)

devnode_devices = list_devnode_devices()
if devnode_devices:
  fallback_device = devnode_devices[(selection_attempt - 1) % len(devnode_devices)]
  print(f"{fallback_device}\tdevnode-round-robin")
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

server_ready_max_attempts=$(((SERVER_READY_TIMEOUT_SECONDS + SERVER_READY_POLL_SECONDS - 1) / SERVER_READY_POLL_SECONDS))
if (( server_ready_max_attempts < 1 )); then
  server_ready_max_attempts=1
fi

server_ready=0

for start_attempt in $(seq 1 "$SERVER_START_RETRIES"); do
  if [[ "$CHIP_COUNT" == "1" ]]; then
    SELECTED_ASCEND_DEVICE_INFO="$(select_ascend_device "$start_attempt" "$NPU_SMI_BIN")"
    if [[ -n "$SELECTED_ASCEND_DEVICE_INFO" ]]; then
      IFS=$'\t' read -r SELECTED_ASCEND_DEVICE SELECTED_ASCEND_DEVICE_SOURCE <<<"$SELECTED_ASCEND_DEVICE_INFO"
      export ASCEND_RT_VISIBLE_DEVICES="$SELECTED_ASCEND_DEVICE"
      export VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE="npu:0"
      echo "selected single-card Ascend device: $ASCEND_RT_VISIBLE_DEVICES (${SELECTED_ASCEND_DEVICE_SOURCE})"
    else
      unset ASCEND_RT_VISIBLE_DEVICES
      unset VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE
      echo "Could not resolve a single-card Ascend device; probing runtime without device scoping"
    fi
  fi

  if wait_for_ascend_runtime_ready; then
    runtime_ready_status=0
  else
    runtime_ready_status=$?
  fi

  if [[ "$runtime_ready_status" -ne 0 ]]; then
    echo "Ascend runtime did not become ready after ${ASCEND_RUNTIME_READY_TIMEOUT_SECONDS}s"
    if [[ "$start_attempt" -lt "$SERVER_START_RETRIES" ]]; then
      echo "Retrying server start after runtime readiness failure in ${SERVER_START_RETRY_DELAY_SECONDS}s (attempt ${start_attempt}/${SERVER_START_RETRIES})"
      sleep "$SERVER_START_RETRY_DELAY_SECONDS"
      continue
    fi
    if [[ "$runtime_ready_status" -eq "$RESOURCE_BUSY_EXIT_CODE" ]]; then
      echo "Detected transient Ascend resource busy state during runtime readiness after exhausting ${SERVER_START_RETRIES} start attempt(s)"
      exit "$RESOURCE_BUSY_EXIT_CODE"
    fi
    exit "$runtime_ready_status"
  fi

  start_server

  for attempt in $(seq 1 "$server_ready_max_attempts"); do
    if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null; then
      server_ready=1
      break 2
    fi

    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "vLLM server exited before becoming ready"
      cat "$SERVER_LOG"
      if server_log_indicates_resource_busy; then
        if [[ "$start_attempt" -lt "$SERVER_START_RETRIES" ]]; then
          echo "Detected transient Ascend resource busy state; retrying server start in ${SERVER_START_RETRY_DELAY_SECONDS}s (attempt ${start_attempt}/${SERVER_START_RETRIES})"
          cleanup
          sleep "$SERVER_START_RETRY_DELAY_SECONDS"
          break
        fi

        echo "Detected transient Ascend resource busy state after exhausting ${SERVER_START_RETRIES} start attempt(s)"
        exit "$RESOURCE_BUSY_EXIT_CODE"
      fi
      exit 1
    fi

    if [[ "$attempt" -eq "$server_ready_max_attempts" ]]; then
      echo "Timed out waiting for vLLM server to become ready after ${SERVER_READY_TIMEOUT_SECONDS}s"
      cat "$SERVER_LOG"
      if server_log_indicates_resource_busy; then
        if [[ "$start_attempt" -lt "$SERVER_START_RETRIES" ]]; then
          echo "Detected transient Ascend resource busy state after timeout; retrying server start in ${SERVER_START_RETRY_DELAY_SECONDS}s (attempt ${start_attempt}/${SERVER_START_RETRIES})"
          cleanup
          sleep "$SERVER_START_RETRY_DELAY_SECONDS"
          break
        fi

        echo "Detected transient Ascend resource busy state after exhausting ${SERVER_START_RETRIES} start attempt(s)"
        exit "$RESOURCE_BUSY_EXIT_CODE"
      fi
      exit 1
    fi

    sleep "$SERVER_READY_POLL_SECONDS"
  done
done

if [[ "$server_ready" != "1" ]]; then
  echo "vLLM server did not become ready after ${SERVER_START_RETRIES} start attempt(s)"
  cat "$SERVER_LOG"
  if server_log_indicates_resource_busy; then
    echo "Detected transient Ascend resource busy state after exhausting ${SERVER_START_RETRIES} start attempt(s)"
    exit "$RESOURCE_BUSY_EXIT_CODE"
  fi
  exit 1
fi

"${VLLM_CLI[@]}" bench serve \
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