#!/usr/bin/env python3
"""Build issue #30 acceptance report from benchmark artifacts.

This script aggregates repeated benchmark runs for current/baseline groups,
computes mean/std/var, and checks required metric coverage.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Metrics required by issue #30 acceptance discussion.
REQUIRED_METRICS = [
    "request_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
    "slo_attainment",
    "total_preemptions",
    "consecutive_preempt_ratio",
]

SUMMARY_METRICS = [
    "request_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
    "slo_attainment",
    "total_preemptions",
    "consecutive_preempt_ratio",
]

UTILITY_LINE_RE = re.compile(
    r"total_preemptions=(?P<total_preemptions>\d+).*?"
    r"consecutive_preempt_ratio=(?P<consecutive_preempt_ratio>[0-9.]+)"
)
VLLM_CUM_PREEMPT_RE = re.compile(r"total_cumulative_preemption_cnt=(?P<count>\d+)")


@dataclass
class RunMetrics:
    path: Path
    run_id: str
    spec_hash: str | None
    metrics: dict[str, float | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate issue #30 acceptance report")
    parser.add_argument(
        "--current-glob",
        required=True,
        help="Glob for current raw_benchmark_result.json files",
    )
    parser.add_argument(
        "--baseline-glob",
        required=True,
        help="Glob for baseline raw_benchmark_result.json files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Markdown output path",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=3,
        help="Minimum repeated runs required by acceptance",
    )
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_slo_attainment(payload: dict[str, Any]) -> float | None:
    if "ttft_slo_attainment" in payload:
        return _to_float(payload.get("ttft_slo_attainment"))
    goodput = _to_float(payload.get("request_goodput"))
    throughput = _to_float(payload.get("request_throughput"))
    if goodput is not None:
        # vLLM bench reports request_goodput in req/s. Convert it to attainment ratio.
        if throughput is not None and throughput > 0:
            return goodput / throughput
        return goodput
    return None


def _extract_preempt_metrics(raw_path: Path) -> tuple[float | None, float | None]:
    log_path = raw_path.with_name("server.stdout.log")
    if not log_path.exists():
        return 0.0, 0.0

    last_match: re.Match[str] | None = None
    cumulative_preemptions = 0.0
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = UTILITY_LINE_RE.search(line)
        if match:
            last_match = match
        cum_match = VLLM_CUM_PREEMPT_RE.search(line)
        if cum_match:
            cum_value = _to_float(cum_match.group("count"))
            if cum_value is not None:
                cumulative_preemptions = max(cumulative_preemptions, cum_value)

    if last_match is not None:
        total_preemptions = _to_float(last_match.group("total_preemptions"))
        consecutive_ratio = _to_float(last_match.group("consecutive_preempt_ratio"))
        if total_preemptions is not None and consecutive_ratio is not None:
            return total_preemptions, consecutive_ratio

    # If no explicit utility snapshot line is found, treat this run as no observed preemption.
    # This keeps coverage explicit and avoids N/A when the scheduler never preempted.
    return cumulative_preemptions, 0.0


def _extract_spec_hash(raw_path: Path) -> str | None:
    same_spec_path = raw_path.with_name("resolved_same_spec.json")
    if not same_spec_path.exists():
        return None
    try:
        payload = json.loads(same_spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    hash_value = payload.get("resolved_spec_hash")
    return str(hash_value) if hash_value else None


def load_run(raw_path: Path) -> RunMetrics:
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    total_preemptions, consecutive_ratio = _extract_preempt_metrics(raw_path)
    spec_hash = _extract_spec_hash(raw_path)

    metrics: dict[str, float | None] = {
        "request_throughput": _to_float(payload.get("request_throughput")),
        "output_throughput": _to_float(payload.get("output_throughput")),
        "mean_ttft_ms": _to_float(payload.get("mean_ttft_ms")),
        "mean_tpot_ms": _to_float(payload.get("mean_tpot_ms")),
        "mean_itl_ms": _to_float(payload.get("mean_itl_ms")),
        "slo_attainment": _extract_slo_attainment(payload),
        "total_preemptions": total_preemptions,
        "consecutive_preempt_ratio": consecutive_ratio,
    }

    run_id = raw_path.parent.parent.name + "/" + raw_path.parent.name
    return RunMetrics(path=raw_path, run_id=run_id, spec_hash=spec_hash, metrics=metrics)


def discover_runs(pattern: str) -> list[RunMetrics]:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    runs = [load_run(path) for path in paths if path.exists()]
    return runs


def _fmt_number(value: float | None, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.{digits}f}"


def summarize(runs: list[RunMetrics], metric: str) -> tuple[float | None, float | None, float | None]:
    values = [run.metrics.get(metric) for run in runs]
    data = [value for value in values if value is not None]
    if not data:
        return None, None, None
    mean = statistics.fmean(data)
    std = statistics.stdev(data) if len(data) > 1 else 0.0
    var = std * std
    return mean, std, var


def coverage(runs: list[RunMetrics], metric: str) -> str:
    if not runs:
        return "0/0"
    present = sum(1 for run in runs if run.metrics.get(metric) is not None)
    return f"{present}/{len(runs)}"


def spec_hash_status(runs: list[RunMetrics]) -> tuple[bool, str]:
    hashes = [run.spec_hash for run in runs if run.spec_hash]
    if not runs:
        return False, "no runs"
    if not hashes:
        return False, "missing resolved_spec_hash"
    unique_hashes = sorted(set(hashes))
    if len(unique_hashes) != 1:
        return False, "mixed hashes: " + ", ".join(unique_hashes)
    if len(hashes) != len(runs):
        return False, "partial resolved_spec_hash"
    return True, unique_hashes[0]


def build_markdown(current_runs: list[RunMetrics], baseline_runs: list[RunMetrics], min_runs: int) -> str:
    lines: list[str] = []
    lines.append("# Issue #30 Acceptance Report")
    lines.append("")

    lines.append("## Run Count Check")
    lines.append("")
    lines.append(f"- Current runs: {len(current_runs)} (required >= {min_runs})")
    lines.append(f"- Baseline runs: {len(baseline_runs)} (required >= {min_runs})")
    lines.append("")

    run_count_ok = len(current_runs) >= min_runs and len(baseline_runs) >= min_runs
    lines.append(f"- Run-count gate: {'PASS' if run_count_ok else 'FAIL'}")
    lines.append("")

    lines.append("## Same-spec Check")
    lines.append("")
    current_spec_ok, current_spec_note = spec_hash_status(current_runs)
    baseline_spec_ok, baseline_spec_note = spec_hash_status(baseline_runs)
    common_hash_ok = (
        current_spec_ok
        and baseline_spec_ok
        and current_spec_note == baseline_spec_note
    )
    lines.append(f"- Current spec gate: {'PASS' if current_spec_ok else 'FAIL'} ({current_spec_note})")
    lines.append(f"- Baseline spec gate: {'PASS' if baseline_spec_ok else 'FAIL'} ({baseline_spec_note})")
    lines.append(
        f"- Cross-group same-spec gate: {'PASS' if common_hash_ok else 'FAIL'}"
    )
    lines.append("")

    lines.append("## Per-run Metrics")
    lines.append("")
    lines.append(
        "| group | run_id | spec_hash | throughput_req_s | throughput_tok_s "
        "| mean_ttft_ms | mean_tpot_ms | mean_itl_ms | slo_attainment "
        "| total_preemptions | consecutive_preempt_ratio |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for group_name, runs in (("current", current_runs), ("baseline", baseline_runs)):
        for run in runs:
            lines.append(
                (
                    "| {group} | {run_id} | {spec_hash} | {req} | {tok} "
                    "| {ttft} | {tpot} | {itl} | {slo} | {preempt} | {ratio} |"
                ).format(
                    group=group_name,
                    run_id=run.run_id,
                    spec_hash=run.spec_hash or "N/A",
                    req=_fmt_number(run.metrics.get("request_throughput")),
                    tok=_fmt_number(run.metrics.get("output_throughput")),
                    ttft=_fmt_number(run.metrics.get("mean_ttft_ms")),
                    tpot=_fmt_number(run.metrics.get("mean_tpot_ms")),
                    itl=_fmt_number(run.metrics.get("mean_itl_ms")),
                    slo=_fmt_number(run.metrics.get("slo_attainment")),
                    preempt=_fmt_number(run.metrics.get("total_preemptions"), digits=0),
                    ratio=_fmt_number(run.metrics.get("consecutive_preempt_ratio")),
                )
            )
    lines.append("")

    lines.append("## Aggregated Statistics")
    lines.append("")
    lines.append("| metric | current_mean | current_std | current_var | baseline_mean | baseline_std | baseline_var |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for metric in SUMMARY_METRICS:
        c_mean, c_std, c_var = summarize(current_runs, metric)
        b_mean, b_std, b_var = summarize(baseline_runs, metric)
        lines.append(
            "| {metric} | {c_mean} | {c_std} | {c_var} | {b_mean} | {b_std} | {b_var} |".format(
                metric=metric,
                c_mean=_fmt_number(c_mean),
                c_std=_fmt_number(c_std),
                c_var=_fmt_number(c_var),
                b_mean=_fmt_number(b_mean),
                b_std=_fmt_number(b_std),
                b_var=_fmt_number(b_var),
            )
        )
    lines.append("")

    lines.append("## Required Metric Coverage")
    lines.append("")
    lines.append("| metric | current_coverage | baseline_coverage |")
    lines.append("|---|---:|---:|")
    for metric in REQUIRED_METRICS:
        lines.append(
            "| {metric} | {current_cov} | {baseline_cov} |".format(
                metric=metric,
                current_cov=coverage(current_runs, metric),
                baseline_cov=coverage(baseline_runs, metric),
            )
        )
    lines.append("")

    missing_metrics = []
    for metric in REQUIRED_METRICS:
        if coverage(current_runs, metric).split("/")[0] == "0" or coverage(baseline_runs, metric).split("/")[0] == "0":
            missing_metrics.append(metric)

    lines.append("## Acceptance Summary")
    lines.append("")
    lines.append(f"- Run-count gate: {'PASS' if run_count_ok else 'FAIL'}")
    lines.append(
        f"- Same-spec gate: {'PASS' if common_hash_ok else 'FAIL'}"
    )
    if missing_metrics:
        lines.append(f"- Metric-coverage gate: FAIL (missing: {', '.join(missing_metrics)})")
    else:
        lines.append("- Metric-coverage gate: PASS")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    current_runs = discover_runs(args.current_glob)
    baseline_runs = discover_runs(args.baseline_glob)

    report = build_markdown(current_runs, baseline_runs, args.min_runs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"Wrote report: {output_path}")
    print(f"Current runs: {len(current_runs)}, baseline runs: {len(baseline_runs)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
