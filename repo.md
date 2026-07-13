# Task Snapshot

## Workspace
- Task directory: `/workspace/.ai-workspace/tasks/2026-07-13-fix-example`
- Main repo: `vllm-hust`
- Plugin repo: `vllm-ascend-hust`
- Working branch: `ws/fix-example`

## What Was Fixed

### 1. ModelScope dependency fallback
- `vllm-hust` falls back to Hugging Face Hub when `VLLM_USE_MODELSCOPE=True` but `modelscope` is unavailable.
- Documentation was updated to explain the fallback behavior and runtime environment requirements.

### 2. ACL runtime import in `camem`
- `vllm_ascend/device_allocator/camem.py` now imports the top-level `acl` extension module instead of `from acl.rt import memcpy`.
- If the worker subprocess starts without Ascend/CANN paths on `sys.path`, `camem` tries common CANN install locations and retries the `acl` import.
- This prevents `EngineCore` subprocess startup from failing when the worker environment is incomplete.

## Validation

### ModelScope / HF
- Real run progressed past ModelScope import failure and continued into Hugging Face loading path.

### ACL / camem
- `python -m py_compile vllm_ascend/device_allocator/camem.py tests/ut/device_allocator/test_camem.py`
- `env TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest tests/ut/device_allocator/test_camem.py -q -k "top_level_acl_module or discovers_acl_site_packages"`
- Result: `2 passed, 12 deselected`

## Commits
- `b31afdd9` `fix acl runtime import in camem`
- `f430530a` `fix acl path discovery for camem`

## How to Recreate in a New Container

1. Copy or mount the same workspace root.
2. Checkout the same branch in the repo worktree:
   ```bash
   git switch ws/fix-example
   git pull --ff-only
   ```
3. If you need Ascend runtime features, source the helper before running:
   ```bash
   source scripts/use_single_ascend_env.sh /usr/local/Ascend/ascend-toolkit/latest
   ```
4. For single-file Python validation of `camem`:
   ```bash
   env TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest tests/ut/device_allocator/test_camem.py -q -k "top_level_acl_module or discovers_acl_site_packages"
   ```

## Notes
- If the new container does not have CANN installed, `camem` will still degrade gracefully by disabling sleep mode.
- The repo also keeps the task state summary in the outer task directory `status.md`.
