# vllm-ascend-hust Roadmap

HUST fork of [vllm-ascend](https://github.com/vllm-project/vllm-ascend) — Ascend NPU adaptation layer for [vllm-hust](https://github.com/vLLM-HUST/vllm-hust).

## Current Version

- **vllm-ascend-hust** 0.19.1 (editable install)
- **vllm-hust** 0.23.1rc1.dev193 (editable install)
- Container: `shuhao-vllm-dev-851` on 8× Ascend 910B3 NPUs

---

## In Progress

### 1. `enable_npugraph_ex` config propagation fix

- **Branch:** `fix/sync-npugraph-ex-to-workers-v2`
- **Commit:** `b19bffd35` on local `main`
- **Problem:** Worker processes silently fall back to `enable_npugraph_ex=True` (torchair) even when the engine sets it to `False` (fusion_pass), because the override is not persisted into `vllm_config.additional_config`.
- **Fix:** `_sync_npugraph_ex_to_additional_config()` helper + call site in `AscendPlatform.check_and_update_config`.
- **PRs:**
    - vLLM-HUST/vllm-ascend-hust #74
    - vllm-project/vllm-ascend #10735 (upstream)
- **Status:** Code committed locally, branch pushed to `fix/sync-npugraph-ex-to-workers-v2`. Needs PR merge.

---

## Short-Term (Q3 2026)

### 2. Stitch + ACL Graph (npugraph_ex) co-existence hardening

- Stitch body-block pinning is invisible to the scheduler's KV cache accounting.
- Patch 6 (proactive eviction in `allocate_slots`) proven on NPU: KV saturation 99.7% → 21.9%, throughput +51.3%.
- **Goal:** upstream the proactive eviction patch into vllm-hust core (currently lives as a monkey-patch / stitch plugin), then validate with ACL Graph enabled (no `--enforce-eager`).

### 3. LFU eviction grace-period validation

- 5-second grace period added to `evict_one()` to protect freshly-registered bodies.
- **Goal:** production validation under sustained multi-user workload; measure cache hit rate stability.

### 4. Sync with upstream vllm-ascend releases

- Rebase vllm-ascend-hust on latest vllm-ascend stable (currently 0.19.x series).
- Cherry-pick upstream fixes for torchair, ACL Graph, and Ascend 910B3 quirks.

---

## Medium-Term (Q4 2026 – Q1 2027)

### 5. Ascend 910C / Atlas 300I A2 support

- Evaluate and adapt vllm-ascend-hust for next-gen Ascend 910C NPUs.
- CANN 9.x compatibility.

### 6. CI benchmark matrix expansion

- Automate multi-model (Qwen3-32B, Qwen3-8B, DeepSeek-V3) × multi-TP (2/4/8) benchmark matrix.
- Integrate segment-reuse (stitch) as a first-class CI benchmark condition.

### 7. Reactive eviction tuning

- Scheduler-level reactive eviction for body blocks under memory pressure.
- Tune eviction thresholds (watermark, grace period) for diverse workload profiles (RAG, code-gen, chat).

---

## Completed

| Item | Commit / PR | Date |
|------|-------------|------|
| Docker lint mirror fix | PR #63 merged | 2026-05 |
| CI engine name canonicalization | `54c82f855` | 2026-05 |
| CI baseline update (v0180/910b2) | `a612f476d` | 2026-05 |
| Dispatch token fix | PR #59 merged | 2026-05 |
| `enable_npugraph_ex` propagation fix | `b19bffd35` / PR #72 / upstream #10735 | 2026-06 |
