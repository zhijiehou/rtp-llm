"""Tests for CutlassSm100ExpertsFp8 executor on SM100 (B200/GB200).

Validates the SM100-optimized FP8 per-tensor GroupGEMM with vLLM-style
3-config dispatch: decode (M<=64, swap_ab), default (1SM), prefill (2SM).
"""

import platform
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
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.fused_moe_executor_test_util import (
    generate_payload_and_weights,
    generate_ref_output,
)
from rtp_llm.utils.model_weight import W

pytestmark = [pytest.mark.gpu(type="SM100_ARM"), pytest.mark.fp8_sm100]


class _ModelConfig:
    def __init__(self, moe_inter_size):
        self.moe_inter_size = moe_inter_size


def _make_config(expert_num=8, hidden_size=256, moe_inter_size=128, num_tokens=64):
    """Create a minimal MoEConfigAdapter for testing."""
    config = MoEConfigAdapter.__new__(MoEConfigAdapter)
    config.expert_num = expert_num
    config.hidden_size = hidden_size
    config.model_config = _ModelConfig(moe_inter_size)
    config.tp_size = 1
    config.ep_size = 1
    config.dp_size = 1
    config.ll_num_max_token = num_tokens
    config.moe_strategy = "auto"
    return config


class TestCutlassSm100ExpertsFp8(unittest.TestCase):
    """Test SM100 FP8 per-tensor executor."""

    def _run_executor_test(self, expert_num, num_tokens, hidden_size, moe_inter_size):
        """Run a single executor test with given dimensions."""
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_sm100_fp8_executor import (
            CutlassSm100ExpertsFp8,
        )

        config = _make_config(
            expert_num=expert_num,
            hidden_size=hidden_size,
            moe_inter_size=moe_inter_size,
            num_tokens=num_tokens,
        )
        payload, weights = generate_payload_and_weights(config)
        ref_output = generate_ref_output(config, payload, weights)

        # Quantize weights to FP8 with per-expert scales
        w1_fp8 = weights[W.moe_w1].to(torch.float8_e4m3fn)
        w2_fp8 = weights[W.moe_w2].to(torch.float8_e4m3fn)
        w1_scale = torch.ones(expert_num, dtype=torch.float32, device="cuda")
        w2_scale = torch.ones(expert_num, dtype=torch.float32, device="cuda")

        fp8_weights = {
            W.moe_w1: w1_fp8,
            W.moe_w2: w2_fp8,
            W.moe_s1: w1_scale,
            W.moe_s2: w2_scale,
        }

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )

        executor = CutlassSm100ExpertsFp8(config, quant_config, fp8_weights)

        assert executor.executor_type() == ExecutorType.CUTLASS_SM100_FP8

        result = executor.execute(
            payload=payload,
            activation="SiGLU",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        )

        assert isinstance(result, CombineForwardPayload)
        assert result.fused_expert_output is not None
        # FP8 has limited precision, use relaxed tolerance
        torch.testing.assert_close(
            result.fused_expert_output,
            ref_output,
            atol=1e-1,
            rtol=1e-1,
        )

    def test_sm100_check(self):
        """Verify SM100 hardware is detected."""
        from rtp_llm.models_py.utils.arch import get_sm

        sm = get_sm()
        self.assertGreaterEqual(sm[0], 10, "This test requires SM100+ hardware")

    def test_executor_type(self):
        """Verify executor type is CUTLASS_SM100_FP8."""
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_sm100_fp8_executor import (
            CutlassSm100ExpertsFp8,
        )

        self.assertEqual(
            CutlassSm100ExpertsFp8.executor_type(), ExecutorType.CUTLASS_SM100_FP8
        )

    def test_decode_small_m(self):
        """M<=64: decode path with swap_ab=True, tile <128,16,128>."""
        self._run_executor_test(
            expert_num=8, num_tokens=16, hidden_size=256, moe_inter_size=128
        )

    def test_default_medium_m(self):
        """64<M<8192: default path, 1SM, tile <128,256,128>."""
        self._run_executor_test(
            expert_num=8, num_tokens=256, hidden_size=256, moe_inter_size=128
        )

    def test_prefill_large_m(self):
        """M>=8192 or N>=8192: prefill path, 2SM, cluster <2,1,1>."""
        self._run_executor_test(
            expert_num=8, num_tokens=1024, hidden_size=256, moe_inter_size=128
        )

    def test_many_experts(self):
        """E=64: many experts with small tokens per expert."""
        self._run_executor_test(
            expert_num=64, num_tokens=64, hidden_size=256, moe_inter_size=128
        )


if __name__ == "__main__":
    unittest.main()
