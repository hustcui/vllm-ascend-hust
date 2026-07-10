#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/ascend-benchmark-leaderboard.yml"
SCRIPT_DIR = REPO_ROOT / ".github/workflows/scripts"
MANAGER_HELPER = REPO_ROOT / "scripts/hust_ascend_manager_helper.sh"
INSTALL_PLUGIN_SCRIPT = REPO_ROOT / "scripts/install_local_ascend_plugin.sh"
INSTALL_DEV_HUB_SCRIPT = SCRIPT_DIR / "install_ascend_benchmark_with_dev_hub.sh"
USE_SINGLE_ASCEND_ENV_SCRIPT = REPO_ROOT / "scripts/use_single_ascend_env.sh"


def test_perfgate_scripts_are_present() -> None:
    for script_name in (
        "perfgate_fetch_baseline.sh",
        "perfgate_stage1_compare.sh",
        "perfgate_stage2_rebase_and_benchmark.sh",
        "perfgate_compare.sh",
        "perfgate_store_baseline.sh",
        "install_ascend_benchmark_with_dev_hub.sh",
        "parse_ascend_comment_command.py",
        "resolve_ascend_benchmark_scenario.py",
        "resolve_perfgate_spec_file.py",
    ):
        assert (SCRIPT_DIR / script_name).is_file()


def test_ascend_benchmark_workflow_wires_two_stage_perfgate() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "PERFGATE_MODE" in workflow
    assert "PERFGATE_SPEC_FILE" in workflow
    assert "SOC_VERSION: ascend910b2" in workflow
    assert (
        "HARDWARE_CHIP_MODEL: ${{ github.event_name == 'workflow_dispatch' && inputs.hardware_chip_model || '910B2' }}"
        in workflow
    )
    assert "Resolve perfgate same-spec file" in workflow
    assert "Resolve main same-spec file" in workflow
    assert "resolve_perfgate_spec_file.py" in workflow
    assert "MAIN_SAME_SPEC_SPEC_FILE:" in workflow
    assert "github.event_name != 'pull_request' && github.event_name != 'issue_comment'" in workflow
    assert '--explicit-chip-model "${HARDWARE_CHIP_MODEL}"' in workflow
    assert '--benchmark-repo "${VLLM_HUST_BENCHMARK_REPO}"' in workflow
    assert '--explicit-same-spec-file ""' in workflow
    assert 'spec_file="${SAME_SPEC_SPEC_FILE:-$MAIN_SAME_SPEC_SPEC_FILE}"' in workflow
    assert "MAIN_BENCH_SCENARIO" in workflow
    assert '--scenario "${MAIN_BENCH_SCENARIO}"' in workflow
    assert '--repo-root "${GITHUB_WORKSPACE}/vllm-hust-benchmark"' in workflow
    assert "docs/official-baselines/perfgate-ascend-qwen25-3b-910b2.json" not in workflow
    assert "docs/official-baselines/perfgate-ascend-qwen25-3b-910b3.json" not in workflow
    assert "perfgate-ascend-qwen25-3b-910b3.json" not in workflow
    assert "VLLM_HUST_BENCHMARK_REF" in workflow
    assert "ref: ${{ env.VLLM_HUST_BENCHMARK_REF }}" in workflow
    assert 'hust_run_pip install -e "${VLLM_HUST_BENCHMARK_REPO}[publish]"' in workflow
    assert "Detect PR fork point" in workflow
    assert "Performance gate - fetch Stage 1 baseline" in workflow
    assert "Performance gate - Stage 1 comparison" in workflow
    assert "Performance gate - Stage 2 trial rebase and benchmark" in workflow
    assert "Performance gate - two-stage comparison" in workflow
    assert "store-main-perfgate-baseline:" in workflow
    assert "Store main perfgate baseline" in workflow
    assert "perfgate_report.md" in workflow
    assert "issue_comment:" in workflow
    assert "Parse Ascend comment command" in workflow
    assert "resolve_ascend_benchmark_scenario.py" in workflow
    assert "github.event_name == 'issue_comment'" in workflow
    assert "benchmark_scenarios:" in workflow
    assert "BENCH_SCENARIOS:" in workflow
    assert "inputs.benchmark_scenarios" in workflow
    assert "vars.VLLM_ASCEND_HUST_PR_BENCHMARK_SCENARIOS" in workflow
    assert "vars.VLLM_ASCEND_HUST_MAIN_BENCHMARK_SCENARIOS" in workflow
    assert "run_ascend_benchmark_scenario_list.sh" in workflow
    assert "steps.resolve-scenario.outputs.BENCH_SCENARIO_COUNT == '1'" in workflow
    assert (
        "(github.event_name == 'pull_request' || github.event_name == 'issue_comment') "
        "&& steps.resolve-scenario.outputs.BENCH_SCENARIO_COUNT == '1'"
    ) in workflow
    assert (
        "github.event_name != 'pull_request' && github.event_name != 'issue_comment' "
        "&& steps.resolve-scenario.outputs.BENCH_SCENARIO_COUNT == '1'"
    ) in workflow
    assert "vars.VLLM_ASCEND_HUST_MAIN_BENCHMARK_SCENARIOS == ''" in workflow
    assert "multi_scenario_results.tsv" in workflow
    assert "Perfgate comparison: `skipped for multi-scenario run" in workflow
    assert "os.environ.get('BENCH_SCENARIO_COUNT', '1') == '1'" in workflow
    assert "timeout-minutes: 60" in workflow
    assert "VLLM_ASCEND_HUST_PUBLISH_BENCHMARK_ON_PR" not in workflow
    assert "github.event_name == 'pull_request' || github.event_name == 'issue_comment'" in workflow
    assert "Checkout dev-hub repo" in workflow
    assert "VLLM_HUST_DEV_HUB_REF" in workflow
    assert "HUST_ASCEND_MANAGER_REF" in workflow
    assert "ref: ${{ env.HUST_ASCEND_MANAGER_REF }}" in workflow
    assert "install_ascend_benchmark_with_dev_hub.sh" in workflow
    assert 'hust_run_pip install "torch==2.9.0"' not in workflow
    assert "scripts/install_local_ascend_plugin.sh" not in workflow
    assert "resolve_cann_major_version()" not in workflow
    assert "vars.VLLM_ASCEND_HUST_BENCHMARK_USE_SUDO || 'auto'" in workflow
    assert "CURRENT_VLLM_CACHE_ROOT: ${{ github.workspace }}/../.hf-cache/vllm" in workflow
    assert "VLLM_ASCEND_HUST_SAME_SPEC_READY_TIMEOUT_SECONDS || '1800'" in workflow
    assert "VLLM_ASCEND_HUST_SAME_SPEC_CLIENT_READY_TIMEOUT_SECONDS || '300'" in workflow
    assert "vars.VLLM_ASCEND_HUST_COMPILE_CUSTOM_KERNELS || 'auto'" in workflow
    assert "VLLM_ASCEND_HUST_STAGE2_DEV_HUB_QUICKSTART_CONDA || '0'" in workflow


def test_schedule_runs_registered_multi_scenario_benchmark_publish() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'cron: "0 19 * * *"' in workflow
    assert "github.event_name == 'schedule'" in workflow
    assert "VLLM_ASCEND_HUST_SCHEDULE_BENCHMARK_SCENARIOS" in workflow
    assert "VLLM_ASCEND_HUST_SCHEDULE_PUBLISH_BENCHMARK != '0'" in workflow
    for scenario in (
        "random-online",
        "sharegpt-online",
        "prefix-repetition-online",
        "random-latency",
        "sharegpt-throughput",
        "sonnet-throughput",
        "instructcoder-online",
        "agent-research-online",
        "visionarena-online",
    ):
        assert scenario in workflow


def test_benchmark_runner_resolves_same_spec_without_random_online_default() -> None:
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")

    assert "SAME_SPEC_SPEC_FILE=${SAME_SPEC_SPEC_FILE:-}" in runner_script
    assert "SAME_SPEC_PR_PREVIEW_COMPAT=${SAME_SPEC_PR_PREVIEW_COMPAT:-1}" in runner_script
    assert "SAME_SPEC_CLIENT_READY_TIMEOUT_SECONDS=${SAME_SPEC_CLIENT_READY_TIMEOUT_SECONDS:-300}" in runner_script
    assert "vllm_hust_benchmark.perfgate_specs resolve" in runner_script
    assert '--scenario "$BENCH_SCENARIO"' in runner_script
    assert '--hardware-chip-model "$HARDWARE_CHIP_MODEL"' in runner_script
    assert '--repo-root "$VLLM_HUST_BENCHMARK_REPO"' in runner_script
    assert "official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json" not in runner_script
    assert 'if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then' in runner_script
    same_spec_block = runner_script[
        runner_script.index('if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then') :
        runner_script.index('else', runner_script.index('if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then'))
    ]
    assert "EFFECTIVE_CONSTRAINTS_FILE=$SAME_SPEC_CONSTRAINTS_FILE" in same_spec_block
    assert "bench_args=()" in same_spec_block
    assert (
        'if [[ "$BENCH_SCENARIO" == "random-online" && "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then'
    ) not in runner_script
    sharegpt_block = runner_script[runner_script.index("    sharegpt-online)") :]
    sharegpt_block = sharegpt_block[: sharegpt_block.index("    *)")]
    assert "BENCH_DATASET_PATH is required for sharegpt-online" in sharegpt_block
    assert 'CLIENT_READY_CHECK_TIMEOUT_SECONDS="$SAME_SPEC_CLIENT_READY_TIMEOUT_SECONDS"' in runner_script
    assert "print_same_spec_server_log_tail" in runner_script
    assert "prepare_same_spec_pr_preview_compat_file()" in runner_script
    assert 'server_parameters["no_enable_chunked_prefill"] = True' in runner_script
    assert 'server_parameters["no_enable_prefix_caching"] = True' in runner_script
    assert 'client_parameters.setdefault("temperature", 0)' in runner_script
    assert 'client_parameters["max_concurrency"] = 1' in runner_script
    assert 'client_parameters["request_rate"] = 1' in runner_script
    assert '"$SAME_SPEC_PR_PREVIEW_COMPAT" == "1"' in runner_script
    assert '"$effective_same_spec_file"' in runner_script
    validation_failure_block = runner_script[
        runner_script.index('if [[ "$validation_status" -ne 0 ]]; then') :
    ]
    validation_failure_block = validation_failure_block[
        : validation_failure_block.index("  fi")
    ]
    assert "print_same_spec_server_log_tail" in validation_failure_block


def test_pull_request_defaults_match_perfgate_spec_size() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "MODEL_NAME: ${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'issue_comment') && "
        "(github.event_name == 'issue_comment' && "
        "needs.issue-comment-command.outputs.model_name || "
        "'Qwen/Qwen2.5-3B-Instruct') || "
        "(github.event_name == 'workflow_dispatch' && inputs.model_name || "
        "'Qwen/Qwen2.5-14B-Instruct') }}"
    ) in workflow
    assert (
        "MODEL_PARAMETERS: ${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'issue_comment') && '3B' || '14B' }}"
    ) in workflow
    assert (
        "MODEL_PRECISION: ${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'issue_comment') && 'BF16' || "
        "(github.event_name == 'workflow_dispatch' && inputs.model_precision || "
        "'FP16') }}"
    ) in workflow
    assert (
        "DTYPE: ${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'issue_comment') && 'bfloat16' || "
        "(github.event_name == 'workflow_dispatch' && inputs.dtype || 'float16') }}"
    ) in workflow
    assert (
        "BENCH_RANDOM_INPUT_LEN: ${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'issue_comment') && '64' || '1024' }}"
    ) in workflow
    assert (
        "BENCH_RANDOM_OUTPUT_LEN: ${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'issue_comment') && '16' || '256' }}"
    ) in workflow


def test_local_ascend_manager_fallback_bootstraps_pip() -> None:
    helper = MANAGER_HELPER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "hust_ensure_python_pip()" in helper
    assert '"${python_bin}" -m ensurepip --upgrade' in helper
    assert "https://bootstrap.pypa.io/get-pip.py" in helper
    assert '"${python_bin}" "${get_pip_script}" --user' in helper
    assert "${CI_HOME:-}" in helper
    assert '"${ci_home}/miniconda3/envs/${env_name}"' in helper
    prepare_step = workflow[workflow.index("Prepare Ascend runtime and install repos") :]
    assert "source scripts/hust_ascend_manager_helper.sh" in prepare_step
    assert prepare_step.index("source scripts/hust_ascend_manager_helper.sh") < prepare_step.index(
        'PYTHON_BIN="$(hust_resolve_python_bin)"'
    )
    assert "export VLLM_HUST_PYTHON_BIN=\"$PYTHON_BIN\"" in workflow
    assert "VLLM_HUST_PYTHON_BIN=$VLLM_HUST_PYTHON_BIN" in workflow
    assert "_hust_ascend_manager_command_needs_pip()" in helper
    assert "--install-python-stack|--install-plugin" in helper
    fallback = helper[helper.index("hust_ascend_manager_run()") :]
    assert 'if _hust_ascend_manager_command_needs_pip "$@"; then' in fallback
    assert 'hust_ensure_python_pip "${python_bin}" || return 1' in fallback
    assert '"${python_bin}" -m hust_ascend_manager.cli "$@"' in fallback


def test_single_ascend_env_falls_back_when_manager_env_fails() -> None:
    single_env = USE_SINGLE_ASCEND_ENV_SCRIPT.read_text(encoding="utf-8")

    assert "manager_env_status=0" in single_env
    assert 'manager_env="$(hust_ascend_manager_run env --shell' in single_env
    assert 'manager_env_status=$?' in single_env
    assert 'if [[ "${manager_env_status}" -eq 0 ]]; then' in single_env
    assert 'eval "${manager_env}"' in single_env
    assert "falling back to local CANN set_env.sh discovery" in single_env
    assert "/usr/local/Ascend/cann-*/set_env.sh" in single_env
    assert '[[ -n "${ASCEND_HOME_PATH:-}" && -n "${ASCEND_OPP_PATH:-}" ]] && python_can_import_tbe' in single_env
    assert 'ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-<unset>}' in single_env


def test_local_plugin_editable_install_bootstraps_build_metadata_deps() -> None:
    install_script = INSTALL_PLUGIN_SCRIPT.read_text(encoding="utf-8")

    assert "import setuptools_scm" in install_script
    assert "import wheel.bdist_wheel" in install_script
    assert 'hust_run_pip install "setuptools-scm>=8"' in install_script
    assert 'hust_run_pip install "wheel"' in install_script
    assert 'hust_run_pip install -e "${PLUGIN_REPO}" --no-build-isolation --no-deps' in install_script


def test_benchmark_prepare_preserves_torch_npu_stack() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    prepare_step = workflow[workflow.index("Prepare Ascend runtime and install repos") :]
    prepare_step = prepare_step[: prepare_step.index("- name: Verify installation")]

    assert "install_ascend_benchmark_with_dev_hub.sh" in prepare_step
    assert "hust_ascend_manager_run setup --non-interactive" not in prepare_step
    assert "ascend-torch-constraints.txt" in prepare_step
    assert "torch==2.10.0" in prepare_step
    assert "torch-npu==2.10.0" in prepare_step
    assert "torchvision==0.25.0" in prepare_step
    assert "torchaudio==2.10.0" in prepare_step
    assert 'hust_run_pip install -c "$torch_constraints"' in prepare_step
    assert 'hust_run_pip install -c "$torch_constraints" -r "$VLLM_HUST_REPO/requirements/common.txt"' in prepare_step
    assert "VLLM_HUST_PYTHON_BIN" in prepare_step


def test_benchmark_runner_auto_disables_sudo_when_unavailable() -> None:
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")

    assert 'if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "auto" ]]; then' in runner_script
    assert "command -v sudo" in runner_script
    assert "Ascend benchmark sudo mode: disabled via auto detection" in runner_script
    assert "command not found" in runner_script[runner_script.index("runtime_ready_log_indicates_sudo_auth_failure") :]


def test_benchmark_server_uses_inferred_max_model_len_by_default() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")
    root_helper = (SCRIPT_DIR / "run_ascend_benchmark_root_helper.sh").read_text(encoding="utf-8")

    assert 'MAX_MODEL_LEN: ""' in workflow
    assert "MAX_MODEL_LEN=${MAX_MODEL_LEN:-}" in runner_script
    assert "max_model_len_args=()" in runner_script
    assert '"${max_model_len_args[@]}"' in runner_script
    assert runner_script.count('"${max_model_len_args[@]}"') == 2
    assert "max_model_len_args=()" in root_helper
    assert '"${max_model_len_args[@]}"' in root_helper
    assert "MAX_MODEL_LEN must be set" not in root_helper


def test_benchmark_server_uses_configurable_eager_and_completions_smoke() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")
    root_helper = (SCRIPT_DIR / "run_ascend_benchmark_root_helper.sh").read_text(encoding="utf-8")

    assert "ASCEND_BENCHMARK_ENFORCE_EAGER:" in workflow
    assert "VLLM_ASCEND_HUST_BENCHMARK_ENFORCE_EAGER || '0'" in workflow
    assert "ASCEND_BENCHMARK_ENFORCE_EAGER=${ASCEND_BENCHMARK_ENFORCE_EAGER:-0}" in runner_script
    assert "ASCEND_BENCHMARK_ENFORCE_EAGER" in runner_script[runner_script.index("SUDO_PRESERVE_ENV_VARS=(") :]
    assert "serve_extra_args=()" in runner_script
    assert "serve_extra_args+=(--enforce-eager)" in runner_script
    assert "run_completions_smoke()" in runner_script
    assert "wait_for_completions_smoke()" in runner_script
    assert "CHAT_SMOKE_TIMEOUT_SECONDS=${CHAT_SMOKE_TIMEOUT_SECONDS:-120}" in runner_script
    assert "CHAT_SMOKE_POLL_SECONDS=${CHAT_SMOKE_POLL_SECONDS:-5}" in runner_script
    assert "CHAT_SMOKE_REQUEST_TIMEOUT_SECONDS=${CHAT_SMOKE_REQUEST_TIMEOUT_SECONDS:-15}" in runner_script
    assert "/v1/completions" in runner_script
    assert "/v1/chat/completions" not in runner_script
    assert "completions_smoke.json" in runner_script
    assert "text.strip()" in runner_script
    assert "completion_tokens > 0" in runner_script
    assert "if wait_for_completions_smoke; then" in runner_script
    assert "Timed out waiting for completions smoke" in runner_script
    assert "--enforce-eager >" not in runner_script
    assert "serve_extra_args=()" in root_helper
    assert "serve_extra_args+=(--enforce-eager)" in root_helper
    assert '"${serve_extra_args[@]}"' in root_helper


def test_same_spec_benchmark_uses_persistent_cache_and_configurable_timeout() -> None:
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")

    assert "SAME_SPEC_READY_TIMEOUT_SECONDS=" in runner_script
    assert "CURRENT_VLLM_CACHE_ROOT=" in runner_script
    assert 'CURRENT_VLLM_CACHE_ROOT="$CURRENT_VLLM_CACHE_ROOT"' in runner_script
    assert 'READY_TIMEOUT_SECONDS="$SAME_SPEC_READY_TIMEOUT_SECONDS"' in runner_script
    assert "SAME_SPEC_READY_TIMEOUT_SECONDS" in runner_script[runner_script.index("SUDO_PRESERVE_ENV_VARS=(") :]


def test_stage2_trial_does_not_publish_benchmark_results() -> None:
    stage2_script = (SCRIPT_DIR / "perfgate_stage2_rebase_and_benchmark.sh").read_text(encoding="utf-8")

    assert "PUBLISH_TO_HF=0" in stage2_script
    assert "PUBLISH_TO_BENCHMARK_REPO=0" in stage2_script
    assert "SYNC_GITHUB_SNAPSHOTS=0" in stage2_script
    assert "BENCHMARK_RESULTS_ROOT" in stage2_script
    assert "install_ascend_benchmark_with_dev_hub.sh" in stage2_script
    assert 'DEV_HUB_QUICKSTART_CONDA="${PERFGATE_STAGE2_DEV_HUB_QUICKSTART_CONDA:-0}"' in stage2_script
    assert "install_local_ascend_plugin.sh" not in stage2_script


def test_dev_hub_install_wrapper_centralizes_custom_kernel_policy() -> None:
    install_script = INSTALL_DEV_HUB_SCRIPT.read_text(encoding="utf-8")

    assert "VLLM_HUST_DEV_HUB_REPO=" in install_script
    assert "ascend-runtime-manager checkout not found" in install_script
    assert "detect_cann_major_version()" in install_script
    assert 'if [[ "$requested" == "auto" ]]; then' in install_script
    assert 'if [[ "$cann_major" == "9" ]]; then' in install_script
    assert "dev-hub-default" in install_script
    assert "COMPILE_CUSTOM_KERNELS=auto resolved to dev-hub default policy for CANN 9" in install_script
    assert "--ascend-lightweight" in install_script
    assert "--ascend-custom-kernels" in install_script
    assert "HUST_DEV_HUB_ASCEND_COMPILE_CUSTOM_KERNELS" in install_script
    assert "HUST_DEV_HUB_SKIP_ASCEND_SYSTEM_APPLY=1" in install_script
    assert 'bash "$VLLM_HUST_DEV_HUB_REPO/scripts/quickstart.sh"' in install_script
    assert "COMPILE_CUSTOM_KERNELS=${COMPILE_CUSTOM_KERNELS:-auto}" in install_script


def test_benchmark_workflow_masks_cross_service_credentials() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "Mask benchmark credentials" in workflow
    assert "::add-mask::" in workflow
    assert '"HF_TOKEN"' in workflow
    assert '"BENCHMARK_REPO_GH_TOKEN"' in workflow
    assert '"BENCHMARK_REPO_SSH_KEY"' in workflow


def test_benchmark_repo_publish_is_gated_and_reported() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")
    sync_script = (SCRIPT_DIR / "sync_benchmark_snapshots_to_github.sh").read_text(encoding="utf-8")

    assert "PUBLISH_TO_BENCHMARK_REPO:" in workflow
    assert "BENCHMARK_REPO_GH_TOKEN:" in workflow
    assert "BENCHMARK_REPO_SSH_KEY:" in workflow
    assert "VLLM_ASCEND_HUST_SYNC_BENCHMARK_SNAPSHOTS_TO_GITHUB || '0'" in workflow
    assert ("github.event_name != 'issue_comment') && secrets.VLLM_HUST_BENCHMARK_GH_TOKEN") in workflow
    assert "L3 Benchmark Repository Publication" in workflow

    assert "PUBLISH_TO_BENCHMARK_REPO=${PUBLISH_TO_BENCHMARK_REPO:-0}" in runner_script
    assert "PUBLISH_TO_BENCHMARK_REPO" in runner_script[runner_script.index("SUDO_PRESERVE_ENV_VARS=(") :]
    assert 'if [[ "$PUBLISH_TO_BENCHMARK_REPO" != "1" ]]; then' in runner_script
    assert 'if [[ "$PUBLISH_TO_BENCHMARK_REPO" == "1" ]]; then' in runner_script
    assert 'elif [[ "$PUBLISH_TO_HF" == "1" ]]; then' not in runner_script
    assert 'elif [[ "$PUBLISH_TO_BENCHMARK_REPO" != "1" ]]; then' in runner_script
    assert 'BENCHMARK_REPO_GH_TOKEN="${BENCHMARK_REPO_GH_TOKEN:-}" \\' in runner_script
    assert 'BENCHMARK_REPO_SSH_KEY="${BENCHMARK_REPO_SSH_KEY:-}" \\' in runner_script

    assert "L3 benchmark repository publication is enabled" in sync_script
    assert "no cross-repository write credential is available" in sync_script
    assert "VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY" in sync_script
    assert "VLLM_HUST_BENCHMARK_GH_TOKEN" in sync_script
    assert "Benchmark repo publish target:" in sync_script
