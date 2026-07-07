# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
USE_SINGLE_ASCEND_ENV = REPO_ROOT / "scripts/use_single_ascend_env.sh"


def test_use_single_ascend_env_falls_back_to_cann_set_env_for_tbe() -> None:
    script = USE_SINGLE_ASCEND_ENV.read_text(encoding="utf-8")

    assert "ensure_cann_tbe_env()" in script
    assert "python_can_import_tbe()" in script
    assert "cann_tbe_python_bin()" in script
    assert "import tbe" in script
    assert "${PYTHON_BIN:-}" in script
    assert "${ASCEND_HOME_PATH:-}/set_env.sh" in script
    assert "${ASCEND_TOOLKIT_HOME:-}/set_env.sh" in script
    assert "${ASCEND_TOOLKIT_LATEST_HOME:-}/set_env.sh" in script
    assert "${CONDA_PREFIX:-}/Ascend/cann/set_env.sh" in script
    assert "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" in script
    assert "/usr/local/Ascend/ascend-toolkit/set_env.sh" in script
    assert 'source "${set_env_file}" || source_status=$?' in script
    assert "ensure_cann_tbe_env || return 1" in script
    assert "Source the correct CANN set_env.sh" in script
