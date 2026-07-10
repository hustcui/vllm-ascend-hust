import argparse
import gc
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

os.environ.setdefault("VLLM_TARGET_DEVICE", "npu")
os.environ.setdefault("VLLM_PLUGINS", "ascend")

from vllm import LLM, SamplingParams


DEFAULT_DATASET = (
    "/root/vllm-ascend-quant-hust/"
    "test-00000-of-00001.parquet"
)


@dataclass
class Sample:
    index: int
    text: str
    token_ids: list[int]
    original_token_count: int
    bucket: str
    was_truncated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline PPL comparison for Qwen3 on Ascend vLLM."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-parquet", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--truncate-mode",
        choices=("skip", "truncate"),
        default="skip",
        help="How to handle samples whose tokenized length exceeds max-model-len.",
    )
    parser.add_argument(
        "--bucket-boundaries",
        type=str,
        default="256,512,1024",
        help="Comma-separated token length bucket upper bounds.",
    )
    parser.add_argument(
        "--kv-cache-dtypes",
        type=str,
        default="auto,kivi_int4",
        help="Comma-separated kv_cache_dtype values to compare.",
    )
    parser.add_argument(
        "--kivi-group-size",
        type=int,
        default=32,
        help="KIVI fake-quant group size when kv_cache_dtype=kivi_int4.",
    )
    parser.add_argument(
        "--kivi-residual-length",
        type=int,
        default=32,
        help="Number of recent full-precision KV tokens kept by KIVI.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code when loading tokenizer/model.",
    )
    parser.add_argument(
        "--no-enforce-eager",
        action="store_true",
        help="Disable eager mode. Not recommended for PPL comparison.",
    )
    parser.add_argument("--disable-sliding-window", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def parse_boundaries(raw: str) -> list[int]:
    boundaries = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"Bucket boundary must be positive, got {value}.")
        boundaries.append(value)
    if not boundaries:
        raise ValueError("At least one bucket boundary is required.")
    return sorted(set(boundaries))


def bucket_name(token_count: int, boundaries: list[int]) -> str:
    lower = 1
    for upper in boundaries:
        if token_count <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f">{boundaries[-1]}"


def load_texts(parquet_path: str, text_column: str | None, max_samples: int) -> list[str]:
    df = pd.read_parquet(parquet_path)
    if text_column is None:
        text_column = next(
            (col for col in ("text", "content", "page") if col in df.columns),
            df.columns[0],
        )
    texts: list[str] = []
    for value in df[text_column].tolist():
        if not isinstance(value, str):
            continue
        text = value.strip()
        if len(text) <= 5 or text.startswith("="):
            continue
        texts.append(text)
        if max_samples > 0 and len(texts) >= max_samples:
            break
    return texts


def prepare_samples(
    texts: list[str],
    tokenizer: AutoTokenizer,
    max_model_len: int,
    min_tokens: int,
    truncate_mode: str,
    boundaries: list[int],
) -> tuple[list[Sample], dict[str, int]]:
    samples: list[Sample] = []
    stats = {
        "raw_texts": len(texts),
        "accepted": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "truncated": 0,
    }
    for idx, text in enumerate(texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        original_len = len(token_ids)
        if original_len < max(min_tokens, 2):
            stats["skipped_too_short"] += 1
            continue
        was_truncated = False
        if original_len > max_model_len:
            if truncate_mode == "skip":
                stats["skipped_too_long"] += 1
                continue
            token_ids = token_ids[:max_model_len]
            was_truncated = True
            stats["truncated"] += 1
            if len(token_ids) < max(min_tokens, 2):
                stats["skipped_too_short"] += 1
                continue
        sample = Sample(
            index=idx,
            text=text,
            token_ids=token_ids,
            original_token_count=original_len,
            bucket=bucket_name(len(token_ids), boundaries),
            was_truncated=was_truncated,
        )
        samples.append(sample)
    stats["accepted"] = len(samples)
    return samples, stats


def make_bucket_stats(boundaries: list[int]) -> dict[str, dict[str, Any]]:
    names = []
    lower = 1
    for upper in boundaries:
        names.append(f"{lower}-{upper}")
        lower = upper + 1
    names.append(f">{boundaries[-1]}")
    return {
        name: {
            "samples": 0,
            "tokens": 0,
            "nll_sum": 0.0,
            "ppl": None,
        }
        for name in names
    }


def build_sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
    )


def release_device_cache() -> None:
    try:
        if hasattr(torch, "accelerator"):
            torch.accelerator.empty_cache()
        elif hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
    except Exception as exc:
        print(f"[WARN] Failed to empty device cache cleanly: {exc}")

    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
    except Exception as exc:
        print(f"[WARN] Failed to synchronize NPU after cache clear: {exc}")


def extract_sample_nll(sample: Sample, prompt_logprobs: Any) -> tuple[float, int]:
    if prompt_logprobs is None:
        raise ValueError("prompt_logprobs is missing.")
    if len(prompt_logprobs) != len(sample.token_ids):
        raise ValueError(
            f"prompt_logprobs length mismatch: got {len(prompt_logprobs)}, "
            f"expected {len(sample.token_ids)}."
        )

    nll_sum = 0.0
    valid_tokens = 0
    for pos in range(1, len(sample.token_ids)):
        chosen_token = sample.token_ids[pos]
        token_logprobs = prompt_logprobs[pos]
        if not token_logprobs:
            raise ValueError(f"Missing prompt logprob at position {pos}.")
        selected = token_logprobs.get(chosen_token)
        if selected is None:
            raise ValueError(
                f"Chosen token id {chosen_token} missing from prompt_logprobs at "
                f"position {pos}."
            )
        nll_sum += -float(selected.logprob)
        valid_tokens += 1
    return nll_sum, valid_tokens


def evaluate_configuration(
    *,
    model_path: str,
    kv_cache_dtype: str,
    samples: list[Sample],
    batch_size: int,
    dtype: str,
    max_model_len: int,
    kivi_group_size: int,
    kivi_residual_length: int,
    trust_remote_code: bool,
    enforce_eager: bool,
    disable_sliding_window: bool,
    gpu_memory_utilization: float,
    boundaries: list[int],
) -> dict[str, Any]:
    print(f"\n[RUN] kv_cache_dtype={kv_cache_dtype}")
    llm = LLM(
        model=model_path,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=1,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        kivi_group_size=kivi_group_size,
        kivi_residual_length=kivi_residual_length,
        disable_sliding_window=disable_sliding_window,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    sampling_params = build_sampling_params()
    bucket_stats = make_bucket_stats(boundaries)
    failures: list[dict[str, Any]] = []
    total_nll_sum = 0.0
    total_tokens = 0
    total_samples = 0

    try:
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            batch_prompts = [sample.token_ids for sample in batch]
            try:
                outputs = llm.generate(batch_prompts, sampling_params)
            except Exception as exc:
                for sample in batch:
                    failures.append(
                        {
                            "index": sample.index,
                            "bucket": sample.bucket,
                            "reason": f"batch_generate_failed: {exc}",
                        }
                    )
                print(
                    f"  batch {start // batch_size}: failed for {len(batch)} "
                    f"samples: {exc}"
                )
                continue

            for sample, output in zip(batch, outputs):
                try:
                    nll_sum, valid_tokens = extract_sample_nll(
                        sample, output.prompt_logprobs
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "index": sample.index,
                            "bucket": sample.bucket,
                            "reason": str(exc),
                        }
                    )
                    continue

                total_nll_sum += nll_sum
                total_tokens += valid_tokens
                total_samples += 1

                bucket_entry = bucket_stats[sample.bucket]
                bucket_entry["samples"] += 1
                bucket_entry["tokens"] += valid_tokens
                bucket_entry["nll_sum"] += nll_sum

            end = min(start + batch_size, len(samples))
            print(
                f"  processed {end}/{len(samples)} samples "
                f"(success={total_samples}, failed={len(failures)})"
            )
    finally:
        del llm
        gc.collect()
        release_device_cache()

    overall_ppl = math.exp(total_nll_sum / total_tokens) if total_tokens else None
    for entry in bucket_stats.values():
        if entry["tokens"] > 0:
            entry["ppl"] = math.exp(entry["nll_sum"] / entry["tokens"])
        del entry["nll_sum"]

    return {
        "kv_cache_dtype": kv_cache_dtype,
        "successful_samples": total_samples,
        "failed_samples": len(failures),
        "tokens": total_tokens,
        "avg_nll": (total_nll_sum / total_tokens) if total_tokens else None,
        "ppl": overall_ppl,
        "bucket_stats": bucket_stats,
        "failures": failures,
    }


def print_summary(prep_stats: dict[str, int], results: list[dict[str, Any]]) -> None:
    print("\n=== Dataset Preparation ===")
    for key, value in prep_stats.items():
        print(f"{key:>20}: {value}")

    print("\n=== PPL Summary ===")
    for result in results:
        print(
            f"{result['kv_cache_dtype']:>12} | "
            f"samples={result['successful_samples']:<4} "
            f"failed={result['failed_samples']:<4} "
            f"tokens={result['tokens']:<7} "
            f"avg_nll={result['avg_nll']:.6f} "
            f"ppl={result['ppl']:.6f}"
            if result["ppl"] is not None and result["avg_nll"] is not None
            else (
                f"{result['kv_cache_dtype']:>12} | samples=0 failed="
                f"{result['failed_samples']}"
            )
        )

        for bucket_name, bucket_entry in result["bucket_stats"].items():
            if bucket_entry["tokens"] == 0:
                continue
            print(
                f"    bucket {bucket_name:>10}: "
                f"samples={bucket_entry['samples']:<4} "
                f"tokens={bucket_entry['tokens']:<7} "
                f"ppl={bucket_entry['ppl']:.6f}"
            )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    boundaries = parse_boundaries(args.bucket_boundaries)
    kv_cache_dtypes = [item.strip() for item in args.kv_cache_dtypes.split(",") if item.strip()]
    if not kv_cache_dtypes:
        raise ValueError("At least one kv_cache_dtype must be provided.")

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=not args.no_trust_remote_code,
    )

    print(f"[INFO] Loading texts from {args.dataset_parquet}")
    texts = load_texts(args.dataset_parquet, args.text_column, args.max_samples)
    print(f"[INFO] Loaded {len(texts)} raw texts.")

    samples, prep_stats = prepare_samples(
        texts=texts,
        tokenizer=tokenizer,
        max_model_len=args.max_model_len,
        min_tokens=args.min_tokens,
        truncate_mode=args.truncate_mode,
        boundaries=boundaries,
    )
    if not samples:
        raise RuntimeError("No valid samples remain after tokenization/filtering.")

    results = []
    for kv_cache_dtype in kv_cache_dtypes:
        result = evaluate_configuration(
            model_path=args.model_path,
            kv_cache_dtype=kv_cache_dtype,
            samples=samples,
            batch_size=args.batch_size,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            kivi_group_size=args.kivi_group_size,
            kivi_residual_length=args.kivi_residual_length,
            trust_remote_code=not args.no_trust_remote_code,
            enforce_eager=not args.no_enforce_eager,
            disable_sliding_window=args.disable_sliding_window,
            gpu_memory_utilization=args.gpu_memory_utilization,
            boundaries=boundaries,
        )
        results.append(result)

    print_summary(prep_stats, results)

    if args.output_json:
        payload = {
            "model_path": args.model_path,
            "dataset_parquet": args.dataset_parquet,
            "max_model_len": args.max_model_len,
            "truncate_mode": args.truncate_mode,
            "prep_stats": prep_stats,
            "results": results,
        }
        output_path = Path(args.output_json)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[INFO] Wrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
