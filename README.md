<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
    <img alt="vllm-ascend" src="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM Ascend Plugin
</h3>

<div align="center">

[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/vllm-project/vllm-ascend)

</div>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>About Ascend</b></a> | <a href="https://docs.vllm.ai/projects/ascend/en/latest/"><b>Documentation</b></a> | <a href="https://slack.vllm.ai"><b>#SIG-Ascend</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support"><b>Users Forum</b></a> | <a href="https://tinyurl.com/vllm-ascend-meeting"><b>Weekly Meeting</b></a> |
</p>

<p align="center">
<a ><b>English</b></a> | <a href="README.zh.md"><b>中文</b></a>
</p>

---
*Latest News* 🔥

- [2026/02] We released the new official version [v0.13.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.13.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.13.0/) to start using vLLM Ascend Plugin on Ascend.
- [2025/12] We released the new official version [v0.11.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.11.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.11.0/) to start using vLLM Ascend Plugin on Ascend.
- [2025/09] We released the new official version [v0.9.1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.9.1)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/large_scale_ep.html) to start deploying large-scale Expert Parallelism (EP) on Ascend.
- [2025/08] We hosted the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/7n8OYNrCC_I9SJaybHA_-Q) with vLLM and Tencent! Please find the meetup slides [here](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF).
- [2025/06] [User stories](https://docs.vllm.ai/projects/ascend/en/latest/community/user_stories/index.html) page is now live! It kicks off with LLaMA-Factory/verl/TRL/GPUStack to demonstrate how vLLM Ascend‌ assists Ascend users in enhancing their experience across fine-tuning, evaluation, reinforcement learning (RL), and deployment scenarios.
- [2025/06] [Contributors](https://docs.vllm.ai/projects/ascend/en/latest/community/contributors.html) page is now live! All contributions deserve to be recorded, thanks for all contributors.
- [2025/05] We've released the first official version [v0.7.3](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3)! We collaborated with the vLLM community to publish a blog post sharing our practice: [Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU](https://blog.vllm.ai/2025/05/12/hardware-plugin.html).
- [2025/03] We hosted the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/VtxO9WXa5fC-mKqlxNUJUQ) with vLLM team! Please find the meetup slides [here](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF).
- [2025/02] vLLM community officially created [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) repo for running vLLM seamlessly on the Ascend NPU.
- [2024/12] We are working with the vLLM community to support [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162).

---

## Overview

vLLM Hust Ascend (`vllm-ascend-hust`) is the localized distribution of the
vLLM Ascend hardware plugin for running `vllm-hust` on the Ascend NPU.

The Python import/module namespace remains `vllm_ascend` for compatibility
with the upstream plugin interface and existing runtime code.

It is the recommended approach for supporting the Ascend backend within the vLLM community. It adheres to the principles outlined in the [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162), providing a hardware-pluggable interface that decouples the integration of the Ascend NPU with vLLM.

By using vLLM Ascend plugin, popular open-source models, including Transformer-like, Mixture-of-Experts (MoE), Embedding, Multi-modal LLMs can run seamlessly on the Ascend NPU.

## Prerequisites

- Hardware: Atlas 800I A2 Inference series, Atlas A2 Training series, Atlas 800I A3 Inference series, Atlas A3 Training series, Atlas 300I Duo (Experimental)
- OS: Linux
- Software:
    - Python >= 3.10, < 3.12
    - CANN == 8.5.0 (Ascend HDK version refers to [here](https://www.hiascend.com/document/detail/zh/canncommercial/83RC2/releasenote/releasenote_0000.html))
    - PyTorch == 2.9.0, torch-npu == 2.9.0
    - vLLM / vLLM Hust (the same compatible version as vllm-ascend-hust)

## Getting Started

Please use the following recommended versions to get started quickly:

| Version    | Release type | Doc                                  |
|------------|--------------|--------------------------------------|
| v0.17.0rc1 | Latest release candidate | See [QuickStart](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html) and [Installation](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) for more details |
| v0.13.0 | Latest stable version | See [QuickStart](https://docs.vllm.ai/projects/ascend/en/v0.13.0/quick_start.html) and [Installation](https://docs.vllm.ai/projects/ascend/en/v0.13.0/installation.html) for more details |

## Local Workspace Helpers

This fork keeps the Python import/module namespace as `vllm_ascend`, while the
distribution package name for packaging and PyPI publication is
`vllm-ascend-hust`.

For the local `vllm-hust` multi-root workspace, Ascend-specific helper scripts
are kept under `scripts/` in this repository rather than in `vllm-hust`.

Common examples:

```bash
# install the local vllm-ascend-hust checkout into the current Python env
bash scripts/install_local_ascend_plugin.sh

# source a single Ascend runtime into the current shell
source scripts/use_single_ascend_env.sh /usr/local/Ascend/ascend-toolkit/latest

# run a small latency benchmark with the same runtime setup
bash scripts/run_ascend_latency_bench.sh /usr/local/Ascend/ascend-toolkit/latest

# bootstrap a local one-command launch flow through hust-ascend-manager
bash scripts/bootstrap_ascend.sh Qwen/Qwen2.5-1.5B-Instruct

# diagnose current Ascend runtime and Python setup
bash scripts/doctor_ascend_env.sh
```

## Qwen2 Runtime Note

`Qwen2ForCausalLM` now defaults to a native rotary fallback on Ascend even when
the runtime stays on the compiled `torch.compile` plus PIECEWISE ACL graph path.
This is narrower than the previous eager-only safety guard and is intended to
avoid the incorrect outputs we observed on the Qwen2 NPU rotary custom-op path
without throwing away the rest of the compiled execution stack.

The local validation that motivated this default used
`Qwen2.5-14B-Instruct` with the shared 3-request sample workload from
`llm-serving-motto-lab`, one warmup batch, and five measured steady-state
batches in the same `vllm-hust` backend environment. Under that setup, the old
eager guard measured `20.78s` mean / `21.44s` p95 batch latency and
`0.144 req/s`; the native-rope fix measured `12.14s` mean / `12.41s` p95 and
`0.247 req/s`.

If you need to disable the fallback for debugging, set
`VLLM_ASCEND_USE_NATIVE_QWEN2_ROPE=0`. If you also need to restore the previous
unsafe compiled behavior for comparison, set
`VLLM_ASCEND_ALLOW_UNSAFE_QWEN2_ACLGRAPH=1` explicitly.

## CI Benchmark Leaderboard

This repository now mirrors the trusted Ascend benchmark publication flow used
in `vllm-hust`.

- Workflow: `.github/workflows/ascend-benchmark-leaderboard.yml`
- Trigger: same-repo pull requests, pushes to `main`, and manual dispatch
- Benchmark source of truth: sibling `vllm-hust-benchmark` repository
- Publish target: Hugging Face dataset snapshots that feed the leaderboard

The workflow checks out a compatible `vllm-hust` baseline, installs the current
`vllm-ascend-hust` plugin checkout on top of it, runs a trusted Ascend serve
benchmark, exports a leaderboard submission artifact, and optionally syncs that
submission plus refreshed leaderboard snapshots to Hugging Face.

Repository variables and secrets follow the `VLLM_ASCEND_HUST_*` prefix, for
example:

- `VLLM_ASCEND_HUST_VLLM_HUST_REF`
- `VLLM_ASCEND_HUST_MAIN_BENCHMARK_SCENARIO`
- `VLLM_ASCEND_HUST_PR_BENCHMARK_SCENARIO`
- `VLLM_ASCEND_HUST_PUBLISH_BENCHMARK_ON_MAIN`
- `VLLM_ASCEND_HUST_PUBLISH_BENCHMARK_ON_PR`
- `VLLM_ASCEND_HUST_BENCHMARK_HF_REPO`
- `VLLM_ASCEND_HUST_LEADERBOARD_URL`
- `VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY` secret to force benchmark checkouts over
  `ssh.github.com:443`
- `HF_TOKEN` secret for trusted HF dataset writes

If `VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY` is configured, the benchmark workflow
rewrites GitHub SSH traffic to `ssh.github.com:443` before any checkout step.
That lets the trusted self-hosted runner avoid the default `github.com:443`
HTTPS clone path when the machine room allows SSH-over-443 more reliably than
HTTPS. If the secret is unset, the workflow falls back to the default GitHub
token + HTTPS checkout behavior.

As in `vllm-hust`, `random-online` runs default to artifact preview only. HF
publication for preview traffic remains gated by
`VLLM_ASCEND_HUST_ALLOW_RANDOM_HF_PUBLISH=1`.

Fork pull requests intentionally do not run the trusted self-hosted benchmark
job automatically. They receive the security-note job only, because the real
benchmark path uses a self-hosted Ascend runner plus trusted dataset-publish
credentials.

Leaderboard display semantics are intentionally repo-scoped and stable:

- displayed `engine`: `vllm-ascend-hust`
- displayed `engine_version`: benchmark target short Git SHA
- artifact `versions.core`: paired `vllm-hust` runtime version
- artifact `versions.backend`: installed `vllm-ascend-hust` package version

Maintainers can still benchmark a fork commit explicitly through
`workflow_dispatch` by setting:

- `ascend_hust_target`: fork target in `owner/repo@ref` form

That manual path checks out the requested fork ref on the trusted runner,
runs the benchmark, and records the actual benchmark target repository/ref/SHA
in the generated submission metadata and benchmark summary.

## CI Maintenance Scope

This fork keeps CI focused on fork-local validation and benchmark publication.

- Upstream repository-maintenance bots such as Dependabot version-bump PRs,
  automatic PR body rewriting, automatic issue labeling, merge-conflict
  labeling, stale-issue cleanup, label-triggered upstream helper jobs, and
  scheduled auto-generated maintenance PRs are intentionally not part of the
  default fork workflow.
- Scheduled upstream patrol workflows such as upstream-main compatibility
  sweeps, routine codecov refresh jobs, and upstream-style periodic performance
  reporting are also not part of the default fork workflow unless this fork
  explicitly decides to operate them.
- If similar automation is needed later, it should be reintroduced explicitly
  against the fork's own maintenance policy instead of inherited unchanged from
  the upstream `vllm-ascend` repository.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for the fork-specific development,
build, and test workflow.

We welcome and value any contributions and collaborations:

- Please let us know if you encounter a bug by [filing an issue](https://github.com/intellistream/vllm-ascend-hust/issues)
- Please use [User forum](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support) for usage questions and help.

## Branch

vllm-ascend-hust keeps a main branch and may carry release branches that track
the compatible upstream vLLM / vLLM Ascend baselines.

- **main**: primary development branch for the localized fork.
- **releases/vX.Y.Z**: optional release branches used when this fork needs to pin to a specific compatible upstream release line.

Below are the maintained branches:

| Branch     | Status       | Note                                 |
|------------|--------------|--------------------------------------|
| main       | Maintained   | CI commitment for vLLM main branch and vLLM v0.17.0 tag   |
| v0.7.1-dev | Unmaintained | Only doc fixes are allowed |
| v0.7.3-dev | Maintained   | CI commitment for vLLM 0.7.3 version, only bug fixes are allowed, and no new release tags anymore. |
| v0.9.1-dev | Maintained   | CI commitment for vLLM 0.9.1 version |
| v0.11.0-dev | Maintained | CI commitment for vLLM 0.11.0 version |
| releases/v0.13.0 | Maintained | CI commitment for vLLM 0.13.0 version |
| rfc/feature-name | Maintained | [Feature branches](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html#feature-branches) for collaboration |

Please refer to [Versioning policy](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html) for more details.

## Weekly Meeting

- vLLM Ascend Weekly Meeting: <https://tinyurl.com/vllm-ascend-meeting>
- Wednesday, 15:00 - 16:00 (UTC+8, [Convert to your timezone](https://dateful.com/convert/gmt8?t=15))

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.
