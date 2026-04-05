"""MoE FP8 SM100 GroupGEMM Performance Benchmark.

Benchmarks FP8 per-tensor MoE GroupGEMM on SM100 (B200/GB200) hardware,
comparing different tile configs and token counts across decode and prefill.

Usage:
    pytest benchmarks/moe_fp8_sm100_benchmark.py -v --timeout=7200

    Or via CI profile:
    pytest --rtp-ci-profile=gb200_fp8_benchmark
"""

import logging
import time
from typing import Dict, List, Tuple

import pytest
import torch

pytestmark = [pytest.mark.gpu(type="SM100_ARM"), pytest.mark.fp8_sm100, pytest.mark.perf]

logger = logging.getLogger(__name__)


def _warmup_and_benchmark(fn, warmup=5, repeat=20):
    """Run warmup iterations, then benchmark with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return {
        "median_ms": sorted(times)[len(times) // 2],
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def _compute_tflops(M, N, K, E, time_ms):
    """Compute TFLOPS for a grouped GEMM."""
    flops = 2.0 * M * N * K * E
    return flops / (time_ms * 1e-3) / 1e12


# DeepSeek-V3 model dimensions
DEEPSEEK_V3_CONFIGS = [
    {"name": "fc1_e8", "E": 8, "N": 7168, "K": 2048},
    {"name": "fc2_e8", "E": 8, "N": 2048, "K": 7168},
    {"name": "fc1_e256", "E": 256, "N": 7168, "K": 2048},
    {"name": "fc2_e256", "E": 256, "N": 2048, "K": 7168},
]

DECODE_TOKEN_COUNTS = [1, 4, 8, 16, 32, 64]
PREFILL_TOKEN_COUNTS = [128, 256, 512, 1024, 2048, 4096]


class TestMoeFp8Sm100Benchmark:
    """Benchmark FP8 per-tensor MoE GroupGEMM on SM100."""

    def _run_cutlass_fp8_benchmark(self, E, N, K, token_counts):
        """Benchmark cutlass_moe_mm_fp8_scaled for given dimensions."""
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import cutlass_moe_mm_fp8_scaled
        from rtp_kernel.fp8_group_gemm import get_cutlass_moe_mm_without_permute_info

        results = []
        for num_tokens in token_counts:
            M = num_tokens
            # Create test tensors
            aq = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            )
            w = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            )
            aq_scale = torch.ones(M, device="cuda", dtype=torch.float32)
            w_scale = torch.ones(E, device="cuda", dtype=torch.float32)
            output = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

            # Construct problem sizes and offsets
            tokens_per_expert = M // E
            expert_offsets = torch.arange(
                0, M + 1, tokens_per_expert, device="cuda", dtype=torch.int32
            )[:E]
            problem_sizes = torch.zeros(E, 3, device="cuda", dtype=torch.int32)
            for i in range(E):
                problem_sizes[i, 0] = tokens_per_expert
                problem_sizes[i, 1] = N
                problem_sizes[i, 2] = K

            a_strides = torch.full((E,), K, device="cuda", dtype=torch.int64)
            b_strides = torch.full((E,), K, device="cuda", dtype=torch.int64)
            c_strides = torch.full((E,), N, device="cuda", dtype=torch.int64)

            def run():
                cutlass_moe_mm_fp8_scaled(
                    output, aq, w, aq_scale, w_scale,
                    expert_offsets, problem_sizes,
                    a_strides, b_strides, c_strides,
                    True, False, M, False,
                )

            try:
                timing = _warmup_and_benchmark(run)
                tflops = _compute_tflops(tokens_per_expert, N, K, E, timing["median_ms"])
                result = {
                    "tokens": num_tokens,
                    "tokens_per_expert": tokens_per_expert,
                    **timing,
                    "tflops": tflops,
                }
                results.append(result)
                logger.info(
                    f"E={E} N={N} K={K} tokens={num_tokens}: "
                    f"{timing['median_ms']:.3f}ms, {tflops:.1f} TFLOPS"
                )
            except Exception as e:
                logger.warning(f"E={E} N={N} K={K} tokens={num_tokens}: FAILED - {e}")
                results.append({"tokens": num_tokens, "error": str(e)})

        return results

    @pytest.mark.parametrize(
        "model_config",
        DEEPSEEK_V3_CONFIGS,
        ids=[c["name"] for c in DEEPSEEK_V3_CONFIGS],
    )
    def test_decode_benchmark(self, model_config):
        """Benchmark decode (small M) scenarios."""
        results = self._run_cutlass_fp8_benchmark(
            E=model_config["E"],
            N=model_config["N"],
            K=model_config["K"],
            token_counts=DECODE_TOKEN_COUNTS,
        )
        # Just ensure it ran without errors
        for r in results:
            assert "error" not in r, f"Failed at tokens={r['tokens']}: {r['error']}"

    @pytest.mark.parametrize(
        "model_config",
        DEEPSEEK_V3_CONFIGS,
        ids=[c["name"] for c in DEEPSEEK_V3_CONFIGS],
    )
    def test_prefill_benchmark(self, model_config):
        """Benchmark prefill (large M) scenarios."""
        results = self._run_cutlass_fp8_benchmark(
            E=model_config["E"],
            N=model_config["N"],
            K=model_config["K"],
            token_counts=PREFILL_TOKEN_COUNTS,
        )
        for r in results:
            assert "error" not in r, f"Failed at tokens={r['tokens']}: {r['error']}"
