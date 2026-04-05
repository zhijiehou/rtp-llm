"""Tests for FlashInferFp8GroupwiseExecutor on SM100 (B200/GB200).

Validates the FlashInfer FP8 groupwise SM100 executor with
group_gemm_fp8_nt_groupwise kernel. Tests use flat 2D token layout
(EP router format) since the executor uses ep_scatter/ep_gather.
"""

import unittest

import pytest
import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.utils.model_weight import W

pytestmark = [pytest.mark.gpu(type="SM100_ARM"), pytest.mark.fp8_sm100]

BLOCK_SIZE = 128


class _ModelConfig:
    def __init__(self, moe_inter_size):
        self.moe_inter_size = moe_inter_size


def _make_config(expert_num=8, hidden_size=256, moe_inter_size=128, ep_size=1):
    """Create a minimal MoEConfigAdapter for testing."""
    config = MoEConfigAdapter.__new__(MoEConfigAdapter)
    config.expert_num = expert_num
    config.hidden_size = hidden_size
    config.model_config = _ModelConfig(moe_inter_size)
    config.tp_size = 1
    config.ep_size = ep_size
    config.ep_rank = 0
    config.dp_size = 1
    config.moe_k = 2
    config.ll_num_max_token = 128
    config.moe_strategy = "auto"
    return config


def _generate_flat_payload(
    num_tokens: int,
    expert_num: int,
    hidden_size: int,
    top_k: int = 2,
    device: str = "cuda",
):
    """Generate flat 2D token payload (EP router format).

    Returns:
        expert_x: [num_tokens, hidden_size] bf16
        expert_topk_ids: [num_tokens, top_k] int32
        expert_topk_weights: [num_tokens, top_k] float32
        expert_num_tokens_cpu: [expert_num] list of int
    """
    expert_x = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)

    # Assign tokens to experts round-robin
    expert_topk_ids = torch.zeros(
        num_tokens, top_k, device=device, dtype=torch.int32
    )
    for i in range(num_tokens):
        for k in range(top_k):
            expert_topk_ids[i, k] = (i * top_k + k) % expert_num

    expert_topk_weights = torch.ones(
        num_tokens, top_k, device=device, dtype=torch.float32
    ) / top_k

    # Count tokens per expert
    num_tokens_per_expert = [0] * expert_num
    for i in range(num_tokens):
        for k in range(top_k):
            eid = expert_topk_ids[i, k].item()
            num_tokens_per_expert[eid] += 1

    # Dummy activation scale (not used directly — ep_scatter quantizes)
    expert_x_scale = torch.ones(
        num_tokens, hidden_size // BLOCK_SIZE, device=device, dtype=torch.float32
    )

    return expert_x, expert_topk_ids, expert_topk_weights, num_tokens_per_expert, expert_x_scale


def _compute_ref_output(
    expert_x, expert_topk_ids, expert_topk_weights,
    w1_bf16, w2_bf16, expert_num, top_k,
):
    """Compute reference output in bf16 (no FP8 quantization).

    This is a simple per-token reference: for each token, for each topk expert,
    compute FC1 → SiLU(gate)*up → FC2, then weighted sum.
    """
    num_tokens, K = expert_x.shape
    _, N2, _ = w1_bf16.shape  # [E, 2*N_inter, K]
    N_inter = N2 // 2

    output = torch.zeros(num_tokens, K, device=expert_x.device, dtype=torch.bfloat16)

    for t in range(num_tokens):
        for k in range(top_k):
            eid = expert_topk_ids[t, k].item()
            weight = expert_topk_weights[t, k].item()
            x = expert_x[t:t+1].float()  # [1, K]
            w1 = w1_bf16[eid].float()  # [2*N_inter, K]
            w2 = w2_bf16[eid].float()  # [K, N_inter]

            fc1 = x @ w1.T  # [1, 2*N_inter]
            gate = fc1[:, :N_inter]
            up = fc1[:, N_inter:]
            gate = gate * torch.sigmoid(gate)
            fc2_in = gate * up  # [1, N_inter]
            fc2_out = fc2_in @ w2.T  # [1, K]
            output[t] += (fc2_out * weight).to(torch.bfloat16).squeeze(0)

    return output


class TestFlashInferFp8GroupwiseExecutor(unittest.TestCase):
    """Test FlashInfer FP8 groupwise SM100 executor."""

    def _run_test(self, expert_num, num_tokens, hidden_size, moe_inter_size, top_k=2):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.flashinfer_fp8_groupwise_executor import (
            FlashInferFp8GroupwiseExecutor,
        )

        config = _make_config(
            expert_num=expert_num,
            hidden_size=hidden_size,
            moe_inter_size=moe_inter_size,
        )
        config.moe_k = top_k

        # Generate flat payload
        (
            expert_x, expert_topk_ids, expert_topk_weights,
            num_tokens_per_expert, expert_x_scale,
        ) = _generate_flat_payload(
            num_tokens=num_tokens,
            expert_num=expert_num,
            hidden_size=hidden_size,
            top_k=top_k,
        )

        # Create bf16 weights for reference
        N = moe_inter_size * 2
        K = hidden_size
        w1_bf16 = (
            torch.randn(expert_num, N, K, device="cuda", dtype=torch.float32) * 0.01
        ).to(torch.bfloat16)
        w2_bf16 = (
            torch.randn(expert_num, K, moe_inter_size, device="cuda", dtype=torch.float32) * 0.01
        ).to(torch.bfloat16)

        # Quantize weights to FP8
        w1_fp8 = w1_bf16.to(torch.float8_e4m3fn)
        w2_fp8 = w2_bf16.to(torch.float8_e4m3fn)

        weights = {
            W.moe_w1: w1_fp8,
            W.moe_w2: w2_fp8,
        }

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )

        executor = FlashInferFp8GroupwiseExecutor(config, quant_config, weights)
        self.assertEqual(executor.executor_type(), ExecutorType.FLASHINFER_FP8_GROUPWISE)

        # Build payload
        expert_num_tokens_tensor = torch.tensor(
            num_tokens_per_expert, dtype=torch.int32, device="cuda"
        )
        payload = ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_scale=expert_x_scale,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=expert_topk_ids,
            expert_topk_weights=expert_topk_weights,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens_tensor,
                expert_num_tokens_cpu=num_tokens_per_expert,
            ),
        )

        result = executor.execute(
            payload=payload,
            activation="SiGLU",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        )

        self.assertIsInstance(result, CombineForwardPayload)
        self.assertIsNotNone(result.fused_expert_output)
        self.assertEqual(result.fused_expert_output.shape, expert_x.shape)

        # Compute reference in bf16 (using bf16 weights, not fp8)
        ref = _compute_ref_output(
            expert_x, expert_topk_ids, expert_topk_weights,
            w1_bf16, w2_bf16, expert_num, top_k,
        )

        # FP8 quantization introduces significant error — use relaxed tolerance
        # Check that the output is in the right ballpark (not NaN/Inf/zero)
        self.assertFalse(torch.isnan(result.fused_expert_output).any())
        self.assertFalse(torch.isinf(result.fused_expert_output).any())

        # Cosine similarity should be reasonable despite FP8 quantization
        cos_sim = torch.nn.functional.cosine_similarity(
            result.fused_expert_output.float().flatten(),
            ref.float().flatten(),
            dim=0,
        )
        self.assertGreater(
            cos_sim.item(), 0.8,
            f"Cosine similarity too low: {cos_sim.item():.4f}"
        )

    # ---- Phase 1: Environment Verification (run first) ----

    def test_phase1_sm100_hardware(self):
        """Phase 1: Verify SM100+ GPU is available."""
        self.assertTrue(torch.cuda.is_available(), "CUDA not available")
        cap = torch.cuda.get_device_capability()
        self.assertGreaterEqual(cap[0], 10, f"SM{cap[0]}{cap[1]} < SM100")
        print(f"GPU: {torch.cuda.get_device_name()}, SM{cap[0]}{cap[1]}")

    def test_phase1_flashinfer_version(self):
        """Phase 1: Verify FlashInfer is installed with correct version."""
        import flashinfer
        version = getattr(flashinfer, "__version__", "unknown")
        print(f"FlashInfer version: {version}")
        self.assertNotEqual(version, "unknown", "FlashInfer version not found")

    def test_phase1_flashinfer_groupwise_api(self):
        """Phase 1: Verify group_gemm_fp8_nt_groupwise API exists."""
        from flashinfer.gemm import group_gemm_fp8_nt_groupwise
        self.assertTrue(callable(group_gemm_fp8_nt_groupwise))
        print("group_gemm_fp8_nt_groupwise: available")

    def test_phase1_rtp_kernel_fp8(self):
        """Phase 1: Verify rtp_kernel FP8 group GEMM is available."""
        try:
            from rtp_kernel.fp8_group_gemm import fp8_grouped_gemm_ptpc
            print("rtp_kernel.fp8_group_gemm: available")
        except ImportError:
            self.skipTest("rtp_kernel not available (optional)")

    def test_phase1_ep_scatter_gather(self):
        """Phase 1: Verify ep_scatter/ep_gather Triton kernels."""
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.flashinfer_fp8_groupwise_executor import (
            FlashInferFp8GroupwiseExecutor,
        )
        # Just verify import succeeds (ep_scatter/ep_gather are imported in executor)
        self.assertTrue(True)
        print("FlashInferFp8GroupwiseExecutor: importable")

    # ---- Phase 2: Unit Tests ----

    def test_executor_type(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.flashinfer_fp8_groupwise_executor import (
            FlashInferFp8GroupwiseExecutor,
        )

        self.assertEqual(
            FlashInferFp8GroupwiseExecutor.executor_type(),
            ExecutorType.FLASHINFER_FP8_GROUPWISE,
        )

    def test_small_tokens(self):
        """Small token count — decode-like scenario, mma_sm=1."""
        self._run_test(
            expert_num=8, num_tokens=16,
            hidden_size=256, moe_inter_size=128,
        )

    def test_medium_tokens(self):
        """Medium token count."""
        self._run_test(
            expert_num=8, num_tokens=128,
            hidden_size=256, moe_inter_size=128,
        )

    def test_large_tokens(self):
        """Large token count — prefill scenario, should use mma_sm=2."""
        self._run_test(
            expert_num=8, num_tokens=512,
            hidden_size=256, moe_inter_size=128,
        )

    def test_many_experts(self):
        """Many experts with few tokens per expert."""
        self._run_test(
            expert_num=64, num_tokens=128,
            hidden_size=256, moe_inter_size=128,
        )


if __name__ == "__main__":
    unittest.main()
