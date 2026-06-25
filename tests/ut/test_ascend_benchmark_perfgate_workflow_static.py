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


def test_perfgate_scripts_are_present() -> None:
    for script_name in (
        "perfgate_fetch_baseline.sh",
        "perfgate_stage1_compare.sh",
        "perfgate_stage2_rebase_and_benchmark.sh",
        "perfgate_compare.sh",
        "perfgate_store_baseline.sh",
        "parse_ascend_comment_command.py",
        "resolve_ascend_benchmark_scenario.py",
    ):
        assert (SCRIPT_DIR / script_name).is_file()


def test_ascend_benchmark_workflow_wires_two_stage_perfgate() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "PERFGATE_MODE" in workflow
    assert "PERFGATE_SPEC_FILE" in workflow
    assert "docs/official-baselines/perfgate-ascend-qwen25-3b-910b3.json" in workflow
    assert "VLLM_HUST_BENCHMARK_REF" in workflow
    assert "ref: ${{ env.VLLM_HUST_BENCHMARK_REF }}" in workflow
    assert 'hust_run_pip install -e "${VLLM_HUST_BENCHMARK_REPO}[publish]"' in workflow
    assert "Detect PR fork point" in workflow
    assert "Performance gate - fetch Stage 1 baseline" in workflow
    assert "Performance gate - Stage 1 comparison" in workflow
    assert "Performance gate - Stage 2 trial rebase and benchmark" in workflow
    assert "Performance gate - two-stage comparison" in workflow
    assert "Store main perfgate baseline" in workflow
    assert "perfgate_report.md" in workflow
    assert "issue_comment:" in workflow
    assert "Parse Ascend comment command" in workflow
    assert "resolve_ascend_benchmark_scenario.py" in workflow
    assert "github.event_name == 'issue_comment'" in workflow
    assert "VLLM_ASCEND_HUST_PUBLISH_BENCHMARK_ON_PR" not in workflow
    assert "github.event_name == 'pull_request' || github.event_name == 'issue_comment'" in workflow
    assert (
        "hust-ascend-manager Python stack reconciliation failed; "
        "falling back to explicit pip installs"
    ) in workflow
    assert "vars.VLLM_ASCEND_HUST_BENCHMARK_USE_SUDO || 'auto'" in workflow


def test_local_ascend_manager_fallback_bootstraps_pip() -> None:
    helper = MANAGER_HELPER.read_text(encoding="utf-8")

    assert "hust_ensure_python_pip()" in helper
    assert '"${python_bin}" -m ensurepip --upgrade' in helper
    assert "https://bootstrap.pypa.io/get-pip.py" in helper
    assert '"${python_bin}" "${get_pip_script}" --user' in helper
    assert "_hust_ascend_manager_command_needs_pip()" in helper
    assert "--install-python-stack|--install-plugin" in helper
    fallback = helper[helper.index("hust_ascend_manager_run()") :]
    assert 'if _hust_ascend_manager_command_needs_pip "$@"; then' in fallback
    assert 'hust_ensure_python_pip "${python_bin}" || return 1' in fallback
    assert '"${python_bin}" -m hust_ascend_manager.cli "$@"' in fallback


def test_local_plugin_editable_install_bootstraps_build_metadata_deps() -> None:
    install_script = INSTALL_PLUGIN_SCRIPT.read_text(encoding="utf-8")

    assert "import setuptools_scm" in install_script
    assert "import wheel.bdist_wheel" in install_script
    assert 'hust_run_pip install "setuptools-scm>=8"' in install_script
    assert 'hust_run_pip install "wheel"' in install_script
    assert 'hust_run_pip install -e "${PLUGIN_REPO}" --no-build-isolation --no-deps' in install_script


def test_benchmark_runner_auto_disables_sudo_when_unavailable() -> None:
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(
        encoding="utf-8"
    )

    assert 'if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "auto" ]]; then' in runner_script
    assert "command -v sudo" in runner_script
    assert "Ascend benchmark sudo mode: disabled via auto detection" in runner_script
    assert "command not found" in runner_script[
        runner_script.index("runtime_ready_log_indicates_sudo_auth_failure") :
    ]


def test_stage2_trial_does_not_publish_benchmark_results() -> None:
    stage2_script = (SCRIPT_DIR / "perfgate_stage2_rebase_and_benchmark.sh").read_text(
        encoding="utf-8"
    )

    assert "PUBLISH_TO_HF=0" in stage2_script
    assert "PUBLISH_TO_BENCHMARK_REPO=0" in stage2_script
    assert "SYNC_GITHUB_SNAPSHOTS=0" in stage2_script
    assert "BENCHMARK_RESULTS_ROOT" in stage2_script
    assert "install_local_ascend_plugin.sh" in stage2_script


def test_benchmark_repo_publish_is_gated_and_reported() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(
        encoding="utf-8"
    )
    sync_script = (SCRIPT_DIR / "sync_benchmark_snapshots_to_github.sh").read_text(
        encoding="utf-8"
    )

    assert "PUBLISH_TO_BENCHMARK_REPO:" in workflow
    assert "BENCHMARK_REPO_GH_TOKEN:" in workflow
    assert "BENCHMARK_REPO_SSH_KEY:" in workflow
    assert "VLLM_ASCEND_HUST_SYNC_BENCHMARK_SNAPSHOTS_TO_GITHUB || '0'" in workflow
    assert (
        "github.event_name != 'issue_comment') && "
        "secrets.VLLM_HUST_BENCHMARK_GH_TOKEN"
    ) in workflow
    assert "L3 Benchmark Repository Publication" in workflow

    assert "PUBLISH_TO_BENCHMARK_REPO=${PUBLISH_TO_BENCHMARK_REPO:-0}" in runner_script
    assert "PUBLISH_TO_BENCHMARK_REPO" in runner_script[
        runner_script.index("SUDO_PRESERVE_ENV_VARS=(") :
    ]
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
