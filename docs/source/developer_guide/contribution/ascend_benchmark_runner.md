# Ascend Benchmark Runner Maintenance

This page documents the runner-local scripts required by the `Ascend Benchmark Leaderboard`
workflow and the fastest way to repair a self-hosted runner when those scripts are missing,
stale, or not executable.

## Required runner-local script paths

The benchmark workflow depends on the following paths inside the runner's local
`vllm-ascend-hust` checkout:

- Repository helper source: `./.github/workflows/scripts/run_ascend_benchmark_root_helper.sh`
- Repository install script: `./scripts/install_ascend_benchmark_root_helper.sh`
- Installed system helper: `/usr/local/bin/run_ascend_benchmark_root_helper.sh`

The workflow compares the installed system helper with the repository helper source before it
starts benchmark execution in sudo mode.

## Install or refresh the runner helper

Run the install script from the runner's local `vllm-ascend-hust` checkout:

```bash
cd /path/to/vllm-ascend-hust
sudo RUNNER_USER=grunner bash scripts/install_ascend_benchmark_root_helper.sh
```

This installs the root helper to `/usr/local/bin/run_ascend_benchmark_root_helper.sh` and
creates the matching sudoers drop-in.

## Quick verification

Use these checks on the runner host when benchmark CI reports a local-script issue:

```bash
cd /path/to/vllm-ascend-hust
test -f .github/workflows/scripts/run_ascend_benchmark_root_helper.sh
test -f scripts/install_ascend_benchmark_root_helper.sh
test -x /usr/local/bin/run_ascend_benchmark_root_helper.sh
cmp -s .github/workflows/scripts/run_ascend_benchmark_root_helper.sh \
  /usr/local/bin/run_ascend_benchmark_root_helper.sh
```

If the final `cmp` command returns non-zero, reinstall the helper from the current checkout.

## Failure signatures and fixes

### Installed helper missing or not executable

Typical job log or PR comment output:

- `installed benchmark root helper is missing or not executable`

Runner host fix:

```bash
cd /path/to/vllm-ascend-hust
sudo RUNNER_USER=grunner bash scripts/install_ascend_benchmark_root_helper.sh
```

### Installed helper is stale

Typical job log or PR comment output:

- `installed benchmark root helper is stale`
- `expected helper source: .../.github/workflows/scripts/run_ascend_benchmark_root_helper.sh`

Runner host fix:

```bash
cd /path/to/vllm-ascend-hust
git pull --ff-only
sudo RUNNER_USER=grunner bash scripts/install_ascend_benchmark_root_helper.sh
```

### Runner checkout script missing

Typical job log or PR comment output:

- `runner-local benchmark root helper source is missing`
- `runner-local benchmark helper install script is missing`

This means the self-hosted runner checkout is incomplete or no longer matches the repository
layout expected by the workflow. Restore or re-sync the local `vllm-ascend-hust` checkout first,
then rerun the install command above.

## Workflow visibility

When the benchmark workflow detects one of these runner-local script problems, it now writes the
diagnosis into the benchmark artifacts and PR comment so the missing path and fix command are
visible without reading the full runner log.