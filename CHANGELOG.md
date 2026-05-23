# Changelog

## Unreleased

### Changed

- Normalized `ASCEND_RT_VISIBLE_DEVICES` in
  `scripts/use_single_ascend_env.sh`: empty or whitespace-only parent values
  are now discarded, non-empty device lists are compacted, and the runtime
  filter falls back to `ASCEND_VISIBLE_DEVICES` when only the generic device
  selection is set. This keeps local shells and trusted benchmark wrappers from
  inheriting an invalid empty runtime device mask.

- Added explicit fork version metadata and git tag conventions for the current
  maintained line: upstream anchors now use `upstream/v...`, fork release tags
  use `v...postN`, and generated builds export `__upstream_version__`,
  `__upstream_commit__`, and `__commit_id__` alongside `__version__`.
- Switched Ascend-side vLLM compatibility checks to upstream version semantics
  so fork suffixes such as `.postN`, `.devM`, and `+gSHA` do not break
  version-gated behavior.
- Renamed the fork's Python distribution name from the upstream-colliding
  `vllm_ascend` package name to `vllm-ascend-hust` across install, uninstall,
  release, and validation entry points while keeping the import namespace as
  `vllm_ascend`.

- Relaxed several optional Ascend imports and registrations so the default text
  serving path used by the same-spec benchmark no longer fails early on missing
  MoE, `torchvision`, FlashLB, or speculative-decoding-only dependencies.
- Deferred MoE op registration to actual MoE models and lazy-loaded the ngram
  proposer / FlashLB code paths, keeping the common `Qwen2.5-14B-Instruct`
  serving path available without pulling in unrelated optional kernels.
- Added the missing thinking-budget batch fields used by newer upstream request
  flows so current benchmark traffic can enter the Ascend runner without input
  batch shape/attribute regressions.

- Defaulted `Qwen2ForCausalLM` on Ascend to a native rotary fallback inside the
  compiled path instead of forcing a full eager fallback. This keeps
  `torch.compile` plus PIECEWISE ACL graph execution enabled while avoiding the
  incorrect outputs previously observed on the NPU rotary custom-op path.
- Documented the local `Qwen2.5-14B-Instruct` small-batch validation used to
  justify that narrower fix: on the shared 3-request sample workload, the old
  eager guard measured `20.78s` mean / `21.44s` p95 batch latency and
  `0.144 req/s`, while the native-rope fix measured `12.14s` mean / `12.41s`
  p95 and `0.247 req/s` under the same backend environment.
- Added optional SSH-over-443 checkout support to the trusted benchmark
  workflow. When the `VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY` secret is set,
  benchmark repository fetches use `ssh.github.com:443`; otherwise the workflow
  keeps the default HTTPS checkout path.
- Added the sibling `vllm-hust-website` checkout to the trusted benchmark
  workflow so preview aggregation can complete instead of failing during
  post-processing.
