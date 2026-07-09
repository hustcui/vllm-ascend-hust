# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ASCEND_910B_CHIP_PATTERN = re.compile(r"910B\d+")


def detect_chip_model_from_text(text: str) -> str:
    normalized = text.upper().replace(" ", "")
    match = ASCEND_910B_CHIP_PATTERN.search(normalized)
    return match.group(0) if match else ""


def detect_chip_model_from_npu_smi(npu_smi_bin: str) -> str:
    if not npu_smi_bin:
        return ""

    for args in (("info",), ("info", "-t", "board")):
        try:
            result = subprocess.run(
                [npu_smi_bin, *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            continue

        chip_model = detect_chip_model_from_text(result.stdout + "\n" + result.stderr)
        if chip_model:
            return chip_model

    return ""


def normalize_chip_model(chip_model: str) -> str:
    return chip_model.upper().replace(" ", "")


def normalize_spec_path(spec_file: str, benchmark_repo: str) -> Path:
    spec_path = Path(spec_file)
    if not spec_path.is_absolute() and benchmark_repo:
        spec_path = Path(benchmark_repo) / spec_path
    spec_path = spec_path.resolve()
    if not spec_path.is_file():
        raise ValueError(f"benchmark spec file not found: {spec_path}")
    return spec_path


def load_shared_perfgate_resolver(benchmark_repo: Path):
    src_dir = benchmark_repo / "src"
    if not src_dir.is_dir():
        raise ValueError(f"benchmark repo src directory not found: {src_dir}")

    for module_name in list(sys.modules):
        if module_name == "vllm_hust_benchmark" or module_name.startswith("vllm_hust_benchmark."):
            sys.modules.pop(module_name, None)
    sys.path.insert(0, str(src_dir))
    try:
        from vllm_hust_benchmark import perfgate_specs
    except Exception as exc:
        raise ValueError(
            "failed to import vllm_hust_benchmark.perfgate_specs from "
            f"{src_dir}; ensure vllm-hust-benchmark contains the shared "
            "perfgate spec resolver"
        ) from exc

    return perfgate_specs.resolve_perfgate_spec_file


def repo_relative_spec_file(spec_path: Path, benchmark_repo: Path) -> str:
    resolved_spec_path = spec_path.resolve()
    resolved_repo = benchmark_repo.resolve()
    try:
        return resolved_spec_path.relative_to(resolved_repo).as_posix()
    except ValueError as exc:
        raise ValueError(f"resolved perfgate spec escapes benchmark repo: {resolved_spec_path}") from exc


def resolve_values(
    *,
    explicit_same_spec_file: str,
    explicit_perfgate_spec_file: str,
    scenario: str,
    explicit_chip_model: str,
    npu_smi_bin: str,
    benchmark_repo: str,
) -> dict[str, str]:
    chip_model = normalize_chip_model(explicit_chip_model)

    if explicit_same_spec_file:
        spec_path = normalize_spec_path(explicit_same_spec_file, benchmark_repo)
        detected_chip = detect_chip_model_from_text(explicit_same_spec_file)
        chip_model = chip_model or normalize_chip_model(detected_chip)
        values = {"SAME_SPEC_SPEC_FILE": str(spec_path)}
        if explicit_perfgate_spec_file:
            values["PERFGATE_SPEC_FILE"] = explicit_perfgate_spec_file
    elif explicit_perfgate_spec_file:
        spec_path = normalize_spec_path(explicit_perfgate_spec_file, benchmark_repo)
        detected_chip = detect_chip_model_from_text(explicit_perfgate_spec_file)
        chip_model = chip_model or normalize_chip_model(detected_chip)
        values = {
            "PERFGATE_SPEC_FILE": explicit_perfgate_spec_file,
            "SAME_SPEC_SPEC_FILE": str(spec_path),
        }
    else:
        chip_model = chip_model or normalize_chip_model(detect_chip_model_from_npu_smi(npu_smi_bin))
        if not chip_model:
            raise ValueError(
                "unable to resolve perfgate spec file because Ascend chip model "
                "is unknown; set HARDWARE_CHIP_MODEL or ensure npu-smi reports "
                "the chip model"
            )
        if not benchmark_repo:
            raise ValueError("VLLM_HUST_BENCHMARK_REPO is required to resolve perfgate spec from the shared registry")

        benchmark_repo_path = Path(benchmark_repo)
        shared_resolver = load_shared_perfgate_resolver(benchmark_repo_path)
        spec_path = Path(
            shared_resolver(
                scenario=scenario,
                hardware_chip_model=chip_model,
                repo_root=benchmark_repo_path,
            )
        )
        if not spec_path.is_absolute():
            spec_path = benchmark_repo_path / spec_path
        if not spec_path.is_file():
            raise ValueError(f"resolved perfgate spec file not found: {spec_path}")

        values = {
            "PERFGATE_SPEC_FILE": repo_relative_spec_file(spec_path, benchmark_repo_path),
            "SAME_SPEC_SPEC_FILE": str(spec_path.resolve()),
        }

    if chip_model:
        values["HARDWARE_CHIP_MODEL"] = chip_model
        values["SOC_VERSION"] = f"ascend{chip_model.lower()}"

    return values


def write_github_env(env_file: str, values: dict[str, str]) -> None:
    if not env_file:
        return

    with Path(env_file).open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve the Ascend perfgate same-spec benchmark file.")
    parser.add_argument(
        "--explicit-same-spec-file",
        default=os.environ.get("SAME_SPEC_SPEC_FILE", "").strip(),
    )
    parser.add_argument(
        "--explicit-perfgate-spec-file",
        default=os.environ.get("PERFGATE_SPEC_FILE", "").strip(),
    )
    parser.add_argument(
        "--scenario",
        default=os.environ.get("BENCH_SCENARIO", "random-online").strip(),
    )
    parser.add_argument(
        "--explicit-chip-model",
        default=os.environ.get("HARDWARE_CHIP_MODEL", "").strip(),
    )
    parser.add_argument(
        "--npu-smi-bin",
        default=os.environ.get("NPU_SMI_BIN", "npu-smi").strip(),
    )
    parser.add_argument(
        "--benchmark-repo",
        default=os.environ.get("VLLM_HUST_BENCHMARK_REPO", "").strip(),
    )
    parser.add_argument(
        "--github-env",
        default=os.environ.get("GITHUB_ENV", "").strip(),
    )
    args = parser.parse_args()

    try:
        values = resolve_values(
            explicit_same_spec_file=args.explicit_same_spec_file,
            explicit_perfgate_spec_file=args.explicit_perfgate_spec_file,
            scenario=args.scenario,
            explicit_chip_model=args.explicit_chip_model,
            npu_smi_bin=args.npu_smi_bin,
            benchmark_repo=args.benchmark_repo,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    write_github_env(args.github_env, values)
    for key, value in values.items():
        print(f"{key}={value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
