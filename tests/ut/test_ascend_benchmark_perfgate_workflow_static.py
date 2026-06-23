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


def test_perfgate_scripts_are_present() -> None:
    for script_name in (
        "perfgate_fetch_baseline.sh",
        "perfgate_stage1_compare.sh",
        "perfgate_stage2_rebase_and_benchmark.sh",
        "perfgate_compare.sh",
        "perfgate_store_baseline.sh",
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
    assert "VLLM_ASCEND_HUST_PUBLISH_BENCHMARK_ON_PR" not in workflow
    assert "github.event_name == 'pull_request' && '0'" in workflow


def test_stage2_trial_does_not_publish_benchmark_results() -> None:
    stage2_script = (SCRIPT_DIR / "perfgate_stage2_rebase_and_benchmark.sh").read_text(
        encoding="utf-8"
    )

    assert "PUBLISH_TO_HF=0" in stage2_script
    assert "PUBLISH_TO_BENCHMARK_REPO=0" in stage2_script
    assert "SYNC_GITHUB_SNAPSHOTS=0" in stage2_script
    assert "BENCHMARK_RESULTS_ROOT" in stage2_script
    assert "install_local_ascend_plugin.sh" in stage2_script
