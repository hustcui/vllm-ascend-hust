#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VLLM_ASCEND_HUST_REPO=${VLLM_ASCEND_HUST_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-${GITHUB_WORKSPACE:-$(cd "$VLLM_ASCEND_HUST_REPO/.." && pwd)}}
VLLM_HUST_REPO=${VLLM_HUST_REPO:-$WORKSPACE_ROOT/vllm-hust}
VLLM_HUST_BENCHMARK_REPO=${VLLM_HUST_BENCHMARK_REPO:-$WORKSPACE_ROOT/vllm-hust-benchmark}
VLLM_HUST_WEBSITE_REPO=${VLLM_HUST_WEBSITE_REPO:-$WORKSPACE_ROOT/vllm-hust-website}
CI_STATE_ROOT=${CI_STATE_ROOT:-$WORKSPACE_ROOT/../vllm-ascend-hust-ci-state}
BENCHMARK_RESULTS_ROOT=${BENCHMARK_RESULTS_ROOT:-$CI_STATE_ROOT/benchmarks/ci}

ASCEND_HUST_TARGET_REPOSITORY=${ASCEND_HUST_TARGET_REPOSITORY:-${GITHUB_REPOSITORY:-vLLM-HUST/vllm-ascend-hust}}
ASCEND_HUST_TARGET_REF=${ASCEND_HUST_TARGET_REF:-${GITHUB_REF_NAME:-detached}}
ASCEND_HUST_TARGET_SHA=${ASCEND_HUST_TARGET_SHA:-${GITHUB_SHA:-$(git -C "$VLLM_ASCEND_HUST_REPO" rev-parse HEAD 2>/dev/null || echo local)}}
ASCEND_HUST_TARGET_SHA_SHORT=$(printf '%s' "$ASCEND_HUST_TARGET_SHA" | cut -c1-8)

RUN_ID=${RUN_ID:-ci-${GITHUB_RUN_ID:+${GITHUB_RUN_ID}-}${GITHUB_RUN_ATTEMPT:-1}-${ASCEND_HUST_TARGET_SHA_SHORT}-simllm-warmcache}
RESULT_ROOT=${RESULT_ROOT:-$BENCHMARK_RESULTS_ROOT/$RUN_ID}
SUBMISSIONS_ROOT=${SUBMISSIONS_ROOT:-$RESULT_ROOT/submissions}
BASELINE_SUBMISSION_DIR=${BASELINE_SUBMISSION_DIR:-$SUBMISSIONS_ROOT/${RUN_ID}-baseline}
SIMLLM_SUBMISSION_DIR=${SIMLLM_SUBMISSION_DIR:-$SUBMISSIONS_ROOT/${RUN_ID}-simllm-enabled}
AGGREGATE_OUTPUT_DIR=${AGGREGATE_OUTPUT_DIR:-$RESULT_ROOT/leaderboard-data}

SPEC_FILE=${1:-${SAME_SPEC_SPEC_FILE:-$VLLM_HUST_BENCHMARK_REPO/docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json}}
CONSTRAINTS_FILE=${CONSTRAINTS_FILE:-$VLLM_HUST_BENCHMARK_REPO/docs/official-baselines/official-ascend-constraints.stub.json}
WARMCACHE_RUNNER=${WARMCACHE_RUNNER:-$VLLM_ASCEND_HUST_REPO/scripts/run_simllm_random_online_warm_cache.sh}

PYTHON_BIN=${PYTHON_BIN:-python3}
CURRENT_RUNTIME_PYTHON=${CURRENT_RUNTIME_PYTHON:-$PYTHON_BIN}
CURRENT_RUNTIME_CWD=${CURRENT_RUNTIME_CWD:-/tmp}
CURRENT_VLLM_CACHE_ROOT=${CURRENT_VLLM_CACHE_ROOT:-$CI_STATE_ROOT/runtime/simllm-warmcache-cache}
CURRENT_SUBMITTER=${CURRENT_SUBMITTER:-${GITHUB_ACTOR:-ci}}
CURRENT_DATA_SOURCE=${CURRENT_DATA_SOURCE:-vllm-ascend-hust-ci-simllm-warmcache}

MODEL_NAME=${MODEL_NAME:-${CURRENT_MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}}
CURRENT_MODEL_PATH=${CURRENT_MODEL_PATH:-$MODEL_NAME}
MODEL_PARAMETERS=${MODEL_PARAMETERS:-14B}
MODEL_PRECISION=${MODEL_PRECISION:-FP16}
MODEL_QUANTIZATION=${MODEL_QUANTIZATION:-}
HARDWARE_CHIP_MODEL=${HARDWARE_CHIP_MODEL:-910B2}
DTYPE=${DTYPE:-float16}
PORT=${PORT:-8021}
BASELINE_SERVER_PORT=${BASELINE_SERVER_PORT:-$PORT}
SIMLLM_SERVER_PORT=${SIMLLM_SERVER_PORT:-$PORT}

RUN_BASELINE=${RUN_BASELINE:-1}
RUN_SIMLLM=${RUN_SIMLLM:-1}
SIMLLM_WARMCACHE_REQUEST_RATE=${SIMLLM_WARMCACHE_REQUEST_RATE:-1}
SIMLLM_WARMCACHE_SEED=${SIMLLM_WARMCACHE_SEED:-0}
SIMLLM_MEASURE_SEED=${SIMLLM_MEASURE_SEED:-$SIMLLM_WARMCACHE_SEED}
SIMLLM_WARMCACHE_PASSES=${SIMLLM_WARMCACHE_PASSES:-1}

PUBLISH_TO_HF=${PUBLISH_TO_HF:-0}
HF_REPO_ID=${HF_REPO_ID:-}
SYNC_GITHUB_SNAPSHOTS=${SYNC_GITHUB_SNAPSHOTS:-0}
SNAPSHOT_TARGET_BRANCH=${SNAPSHOT_TARGET_BRANCH:-main}
SNAPSHOT_MAX_PUSH_ATTEMPTS=${SNAPSHOT_MAX_PUSH_ATTEMPTS:-3}
SNAPSHOT_PUSH_RETRY_SECONDS=${SNAPSHOT_PUSH_RETRY_SECONDS:-10}
GIT_COMMITTER_NAME=${GIT_COMMITTER_NAME:-vLLM-HUST benchmark bot}
GIT_COMMITTER_EMAIL=${GIT_COMMITTER_EMAIL:-benchmark-bot@vllm-hust.local}

if [[ "$PYTHON_BIN" != */* ]]; then
  PYTHON_BIN=$(command -v "$PYTHON_BIN" || true)
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN:-unset}" >&2
  exit 2
fi
if [[ "$CURRENT_RUNTIME_PYTHON" != */* ]]; then
  CURRENT_RUNTIME_PYTHON=$(command -v "$CURRENT_RUNTIME_PYTHON" || true)
fi
if [[ -z "$CURRENT_RUNTIME_PYTHON" || ! -x "$CURRENT_RUNTIME_PYTHON" ]]; then
  echo "CURRENT_RUNTIME_PYTHON is not executable: ${CURRENT_RUNTIME_PYTHON:-unset}" >&2
  exit 2
fi

required_submission_files=(leaderboard_manifest.json run_leaderboard.json)
required_snapshot_files=(last_updated.json leaderboard_single.json leaderboard_multi.json leaderboard_compare.json)

usage() {
  cat >&2 <<EOF
Usage: $0 [random-online-official-spec.json]

Runs the SimLLM warm-cache A/B benchmark and publishes two CI submissions:
  - ${RUN_ID}-baseline
  - ${RUN_ID}-simllm-enabled

Set PUBLISH_TO_HF=1 to upload through sync-submission-to-hf.
Set SYNC_GITHUB_SNAPSHOTS=1 to commit submissions and snapshots into vllm-hust-benchmark.
EOF
}

require_file() {
  local path=$1
  local description=$2
  if [[ ! -f "$path" ]]; then
    echo "$description not found: $path" >&2
    exit 2
  fi
}

require_dir() {
  local path=$1
  local description=$2
  if [[ ! -d "$path" ]]; then
    echo "$description not found: $path" >&2
    exit 2
  fi
}

validate_submission_dir() {
  local submission_dir=$1
  local label=$2
  for file_name in "${required_submission_files[@]}"; do
    if [[ ! -f "$submission_dir/$file_name" ]]; then
      echo "$label submission is missing $file_name: $submission_dir" >&2
      exit 2
    fi
  done
}

validate_raw_result_file() {
  local result_file=$1
  local label=$2
  "$PYTHON_BIN" - "$result_file" "$label" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
label = sys.argv[2]
if not path.is_file():
    raise SystemExit(f"{label} raw benchmark result not found: {path}")

payload = json.loads(path.read_text())
completed = int(payload.get("completed") or payload.get("successful_requests") or 0)
failed = int(payload.get("failed") or payload.get("failed_requests") or 0)
if completed <= 0:
    raise SystemExit(f"{label} benchmark completed no requests: completed={completed}")
if failed != 0:
    raise SystemExit(
        f"{label} benchmark contains failed requests: completed={completed} failed={failed}"
    )
print(f"{label} raw benchmark validated: completed={completed} failed={failed}")
PY
}

normalize_submission_dir() {
  local submission_dir=$1
  "$PYTHON_BIN" - "$submission_dir/run_leaderboard.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
for section, key in (
    ("hardware", "chip_model"),
    ("environment", "hardware_chip_model"),
    ("same_spec", "hardware_chip_model"),
):
    obj = data.get(section)
    if isinstance(obj, dict) and obj.get(key) == "Ascend 910B2":
        obj[key] = "910B2"
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
PY
}

copy_submission_dir() {
  local source_dir=$1
  local target_dir=$2
  local label=$3
  validate_submission_dir "$source_dir" "$label"
  normalize_submission_dir "$source_dir"
  rm -rf "$target_dir"
  mkdir -p "$target_dir"
  for file_name in "${required_submission_files[@]}"; do
    cp "$source_dir/$file_name" "$target_dir/$file_name"
  done
  echo "[$label] submission: $target_dir"
}

publish_local_snapshots() {
  mkdir -p "$AGGREGATE_OUTPUT_DIR"
  "$PYTHON_BIN" -m vllm_hust_benchmark.cli publish-website \
    --source-dir "$SUBMISSIONS_ROOT" \
    --output-dir "$AGGREGATE_OUTPUT_DIR" \
    --execute
  echo "leaderboard snapshots: $AGGREGATE_OUTPUT_DIR"
}

publish_to_hf() {
  if [[ -z "$HF_REPO_ID" ]]; then
    echo "HF_REPO_ID must be set when PUBLISH_TO_HF=1" >&2
    exit 2
  fi

  "$PYTHON_BIN" -m vllm_hust_benchmark.cli sync-submission-to-hf \
    --submission-dir "$BASELINE_SUBMISSION_DIR" \
    --submission-dir "$SIMLLM_SUBMISSION_DIR" \
    --aggregate-output-dir "$AGGREGATE_OUTPUT_DIR" \
    --repo-id "$HF_REPO_ID" \
    --submissions-prefix submissions-auto \
    --commit-message "chore: sync SimLLM warm-cache benchmark from vllm-ascend-hust $RUN_ID (${ASCEND_HUST_TARGET_REPOSITORY}@${ASCEND_HUST_TARGET_REF}:${ASCEND_HUST_TARGET_SHA_SHORT})" \
    --execute
}

prepare_github_snapshot_commit() {
  local target_baseline_dir=$VLLM_HUST_BENCHMARK_REPO/submissions/$(basename "$BASELINE_SUBMISSION_DIR")
  local target_simllm_dir=$VLLM_HUST_BENCHMARK_REPO/submissions/$(basename "$SIMLLM_SUBMISSION_DIR")
  local snapshot_output_dir=$VLLM_HUST_BENCHMARK_REPO/leaderboard-data/snapshots

  git -C "$VLLM_HUST_BENCHMARK_REPO" fetch origin "$SNAPSHOT_TARGET_BRANCH"
  git -C "$VLLM_HUST_BENCHMARK_REPO" switch "$SNAPSHOT_TARGET_BRANCH"
  git -C "$VLLM_HUST_BENCHMARK_REPO" reset --hard "origin/$SNAPSHOT_TARGET_BRANCH"

  mkdir -p "$target_baseline_dir" "$target_simllm_dir" "$snapshot_output_dir"
  for file_name in "${required_submission_files[@]}"; do
    cp "$BASELINE_SUBMISSION_DIR/$file_name" "$target_baseline_dir/$file_name"
    cp "$SIMLLM_SUBMISSION_DIR/$file_name" "$target_simllm_dir/$file_name"
  done

  for file_name in "${required_snapshot_files[@]}"; do
    rm -f "$snapshot_output_dir/$file_name"
  done

  "$PYTHON_BIN" -m vllm_hust_benchmark.cli publish-website \
    --source-dir "$VLLM_HUST_BENCHMARK_REPO/submissions" \
    --output-dir "$snapshot_output_dir" \
    --execute

  for file_name in "${required_snapshot_files[@]}"; do
    if [[ ! -f "$snapshot_output_dir/$file_name" ]]; then
      echo "missing generated snapshot file: $snapshot_output_dir/$file_name" >&2
      exit 2
    fi
  done

  git -C "$VLLM_HUST_BENCHMARK_REPO" add \
    "submissions/$(basename "$BASELINE_SUBMISSION_DIR")" \
    "submissions/$(basename "$SIMLLM_SUBMISSION_DIR")" \
    leaderboard-data/snapshots

  if git -C "$VLLM_HUST_BENCHMARK_REPO" diff --cached --quiet; then
    return 1
  fi

  git -C "$VLLM_HUST_BENCHMARK_REPO" commit \
    -m "chore: sync SimLLM warm-cache benchmark snapshots $RUN_ID"
}

sync_github_snapshots() {
  if [[ "${GITHUB_ACTIONS:-}" != "true" && "${ALLOW_LOCAL_GIT_RESET:-0}" != "1" ]]; then
    echo "refusing to reset a local checkout outside GitHub Actions; set ALLOW_LOCAL_GIT_RESET=1 to override" >&2
    exit 2
  fi

  require_dir "$VLLM_HUST_BENCHMARK_REPO/.git" "benchmark git checkout"
  git -C "$VLLM_HUST_BENCHMARK_REPO" config user.name "$GIT_COMMITTER_NAME"
  git -C "$VLLM_HUST_BENCHMARK_REPO" config user.email "$GIT_COMMITTER_EMAIL"

  for attempt in $(seq 1 "$SNAPSHOT_MAX_PUSH_ATTEMPTS"); do
    if ! prepare_github_snapshot_commit; then
      echo "GitHub leaderboard snapshots already include SimLLM warm-cache submissions for $RUN_ID"
      return 0
    fi

    local snapshot_commit
    snapshot_commit=$(git -C "$VLLM_HUST_BENCHMARK_REPO" rev-parse HEAD)
    if git -C "$VLLM_HUST_BENCHMARK_REPO" push origin "HEAD:$SNAPSHOT_TARGET_BRANCH"; then
      echo "Pushed GitHub leaderboard snapshots to vllm-hust-benchmark@$SNAPSHOT_TARGET_BRANCH: $snapshot_commit"
      return 0
    fi

    echo "snapshot push failed; retrying with fresh origin/$SNAPSHOT_TARGET_BRANCH in ${SNAPSHOT_PUSH_RETRY_SECONDS}s (attempt $attempt/$SNAPSHOT_MAX_PUSH_ATTEMPTS)" >&2
    sleep "$SNAPSHOT_PUSH_RETRY_SECONDS"
  done

  echo "failed to push GitHub leaderboard snapshots after $SNAPSHOT_MAX_PUSH_ATTEMPTS attempts" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_dir "$VLLM_ASCEND_HUST_REPO" "vllm-ascend-hust repo"
require_dir "$VLLM_HUST_REPO" "vllm-hust repo"
require_dir "$VLLM_HUST_BENCHMARK_REPO" "vllm-hust-benchmark repo"
require_file "$SPEC_FILE" "same-spec benchmark spec"
require_file "$CONSTRAINTS_FILE" "constraints file"
require_file "$WARMCACHE_RUNNER" "SimLLM warm-cache runner"

export PYTHONPATH="$VLLM_HUST_BENCHMARK_REPO/src:$VLLM_ASCEND_HUST_REPO:$VLLM_HUST_REPO${PYTHONPATH:+:$PYTHONPATH}"
LOCAL_NO_PROXY="localhost,127.0.0.1,::1,0.0.0.0"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$LOCAL_NO_PROXY"
export no_proxy="${no_proxy:+$no_proxy,}$LOCAL_NO_PROXY"

current_vllm_hust_commit=$(git -C "$VLLM_HUST_REPO" rev-parse HEAD 2>/dev/null || true)

rm -rf "$RESULT_ROOT"
mkdir -p "$RESULT_ROOT" "$SUBMISSIONS_ROOT" "$CURRENT_VLLM_CACHE_ROOT"

echo "== SimLLM warm-cache benchmark CI =="
echo "workspace: $WORKSPACE_ROOT"
echo "ascend repo: $VLLM_ASCEND_HUST_REPO"
echo "benchmark repo: $VLLM_HUST_BENCHMARK_REPO"
echo "run id: $RUN_ID"
echo "result root: $RESULT_ROOT"
echo "spec file: $SPEC_FILE"
echo "publish to hf: $PUBLISH_TO_HF"
echo "sync github snapshots: $SYNC_GITHUB_SNAPSHOTS"

env \
  VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
  CURRENT_RUNTIME_CWD="$CURRENT_RUNTIME_CWD" \
  CURRENT_RUNTIME_PYTHON="$CURRENT_RUNTIME_PYTHON" \
  CURRENT_MODEL_NAME="$MODEL_NAME" \
  CURRENT_MODEL_PATH="$CURRENT_MODEL_PATH" \
  CURRENT_MODEL_PARAMETERS="$MODEL_PARAMETERS" \
  CURRENT_MODEL_PRECISION="$MODEL_PRECISION" \
  CURRENT_MODEL_QUANTIZATION="$MODEL_QUANTIZATION" \
  CURRENT_HARDWARE_CHIP_MODEL="$HARDWARE_CHIP_MODEL" \
  CURRENT_DTYPE="$DTYPE" \
  CURRENT_VLLM_HUST_REPO="$VLLM_HUST_REPO" \
  CURRENT_VLLM_ASCEND_HUST_REPO="$VLLM_ASCEND_HUST_REPO" \
  CURRENT_VLLM_CACHE_ROOT="$CURRENT_VLLM_CACHE_ROOT" \
  CURRENT_GITHUB_REPOSITORY="vLLM-HUST/vllm-hust" \
  CURRENT_GITHUB_REF="${VLLM_HUST_REF:-main}" \
  CURRENT_GIT_COMMIT="$current_vllm_hust_commit" \
  CURRENT_PLUGIN_ENGINE="vllm-ascend-hust" \
  CURRENT_PLUGIN_GITHUB_REPOSITORY="$ASCEND_HUST_TARGET_REPOSITORY" \
  CURRENT_PLUGIN_GITHUB_REF="$ASCEND_HUST_TARGET_REF" \
  CURRENT_PLUGIN_GIT_COMMIT="$ASCEND_HUST_TARGET_SHA" \
  CURRENT_SUBMITTER="$CURRENT_SUBMITTER" \
  CURRENT_DATA_SOURCE="$CURRENT_DATA_SOURCE" \
  RESULT_DIR="$RESULT_ROOT" \
  BASELINE_DIR="$RESULT_ROOT/baseline-disabled" \
  SIMLLM_DIR="$RESULT_ROOT/enabled-warm-cache" \
  RUN_BASELINE="$RUN_BASELINE" \
  RUN_SIMLLM="$RUN_SIMLLM" \
  RUN_ID="$RUN_ID" \
  BASELINE_SERVER_PORT="$BASELINE_SERVER_PORT" \
  SIMLLM_SERVER_PORT="$SIMLLM_SERVER_PORT" \
  CONSTRAINTS_FILE="$CONSTRAINTS_FILE" \
  SIMLLM_WARMCACHE_REQUEST_RATE="$SIMLLM_WARMCACHE_REQUEST_RATE" \
  SIMLLM_WARMCACHE_SEED="$SIMLLM_WARMCACHE_SEED" \
  SIMLLM_MEASURE_SEED="$SIMLLM_MEASURE_SEED" \
  SIMLLM_WARMCACHE_PASSES="$SIMLLM_WARMCACHE_PASSES" \
  bash "$WARMCACHE_RUNNER" "$SPEC_FILE"

if [[ "$RUN_BASELINE" == "1" ]]; then
  validate_raw_result_file "$RESULT_ROOT/baseline-disabled/raw_benchmark_result.json" "baseline"
fi
if [[ "$RUN_SIMLLM" == "1" ]]; then
  validate_raw_result_file "$RESULT_ROOT/enabled-warm-cache/raw_benchmark_result.json" "simllm-enabled"
fi

copy_submission_dir "$RESULT_ROOT/baseline-disabled/submission" "$BASELINE_SUBMISSION_DIR" "baseline"
copy_submission_dir "$RESULT_ROOT/enabled-warm-cache/submission" "$SIMLLM_SUBMISSION_DIR" "simllm-enabled"

publish_local_snapshots

if [[ "$PUBLISH_TO_HF" == "1" ]]; then
  publish_to_hf
fi

if [[ "$SYNC_GITHUB_SNAPSHOTS" == "1" ]]; then
  sync_github_snapshots
fi

echo "RUN_ID=$RUN_ID"
echo "BASELINE_SUBMISSION_DIR=$BASELINE_SUBMISSION_DIR"
echo "SIMLLM_SUBMISSION_DIR=$SIMLLM_SUBMISSION_DIR"
echo "AGGREGATE_OUTPUT_DIR=$AGGREGATE_OUTPUT_DIR"
