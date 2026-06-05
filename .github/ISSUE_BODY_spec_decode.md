## Summary

Speculative decoding is **fundamentally non-functional** in the current vllm-ascend build (v0.19.x, based on upstream vLLM 0.18.0). Multiple speculation methods have been tested and all crash at runtime on Ascend NPU hardware. This is not a single-method limitation but a systemic incompatibility in the Ascend verification path.

## Affected Methods

| Method | Status | Failure Mode |
|--------|--------|--------------|
| `draft_model` | :x: Crashes | `AscendDraftModelProposer` missing `update_stream` implementation — RuntimeError at engine init |
| `ngram` | :x: Crashes | Proposer runs on CPU but crashes during interaction with the Ascend verification/scoring path |
| `suffix` | :warning: Expected broken | Same verification-path incompatibility as ngram (not independently tested) |
| `eagle` / `eagle3` / `mtp` | :grey_question: Untested | Separate Ascend-native code path (`AscendEagleProposer`) — may work but unvalidated |

## Environment

- **Hardware:** Huawei Atlas 910B (Ascend NPU)
- **vllm-ascend-hust:** v0.19.1rc1 (commit `1458891fc`, main branch)
- **Upstream vLLM:** 0.18.0
- **OS:** openEuler 22.03
- **CANN:** 8.x

## Steps to Reproduce

### 1. Draft Model Speculation (crashes at engine init)

```bash
vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 4 \
  --speculative-config '{"method": "draft_model", "model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5}'
```

**Expected:** Engine starts with draft model speculation enabled.  
**Actual:** RuntimeError — `AscendDraftModelProposer` does not implement `update_stream`.

### 2. Ngram-Based Speculation (crashes during inference)

```bash
vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 4 \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'
```

**Expected:** Engine runs with ngram speculation (CPU-based proposer, Ascend verification).  
**Actual:** Crash during the Ascend verification/scoring step. The ngram proposer itself runs on CPU, but the token verification path on the NPU is broken.

## Root Cause Analysis

The speculative decoding verification step (where proposed tokens are scored by the target model in a single forward pass) appears to have a fundamental incompatibility with the Ascend execution path. This affects all methods that rely on the standard verification pipeline — only EAGLE-based methods bypass this through a completely separate Ascend-native code path (`AscendEagleProposer`).

Key areas in the codebase:
- `vllm_ascend/spec_decode/__init__.py` — method dispatch
- `vllm_ascend/worker/v2/spec_decode/` — Ascend-specific speculation workers

## Workaround

**Disable speculative decoding entirely.** Do not pass `--speculative-config` or `speculative_config` to the engine:

```bash
# Working configuration (no speculation)
vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90
```

## Impact

- Users cannot leverage speculative decoding for latency reduction on Ascend hardware
- Documentation currently lists ngram as a "workaround" for the draft model limitation — this is misleading since ngram also fails
- Performance-sensitive deployments must rely solely on batching and model parallelism for throughput

## Requested Action

1. Fix the Ascend verification path to support standard speculative decoding proposals
2. Alternatively, document the limitation prominently and remove ngram as a recommended workaround
3. Clarify which EAGLE-based methods (if any) are validated on current hardware

## Labels

`bug`, `ascend`, `speculative-decoding`, `P1`
