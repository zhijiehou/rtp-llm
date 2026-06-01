from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    configure_deep_gemm_num_sms,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
    DeepGemmHybridExecutor,
)
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128


class DeepGemmContiguousExecutorV2(DeepGemmHybridExecutor):
    """Contiguous GEMM executor using psum layout (DeepGEMM >= 2.5.0).

    Uses use_psum_layout=True to pass psum directly to the GEMM kernel.
    No ep_scatter, no ep_gather, no m_indices construction, no D2H.
    """

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.DEEPGEMM_CONTINUOUS

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(resolver.is_bf16(config))
        checker.check(has_deep_gemm())
        checker.check(get_sm()[0] >= 9)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None
        assert payload.expert_x_scale is not None
        assert payload.expert_tokens_meta is not None
        assert payload.expert_tokens_meta.expert_psum is not None

        import deep_gemm

        hidden_states_fp8 = payload.expert_x
        hidden_states_scale = payload.expert_x_scale
        psum = payload.expert_tokens_meta.expert_psum

        total_tokens = hidden_states_fp8.shape[0]
        _, K = hidden_states_fp8.size()
        N = self.w13_weight.size(1)
        device = hidden_states_fp8.device

        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            # Gate+Up GEMM with psum layout
            gateup_output = torch.empty(
                (total_tokens, N), device=device, dtype=torch.bfloat16,
            )
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (hidden_states_fp8, hidden_states_scale),
                self.w13_weight_fp8,
                gateup_output,
                psum,
                use_psum_layout=True,
            )

            # SiLU activation
            down_input = torch.empty(
                (total_tokens, N // 2), device=device, dtype=torch.bfloat16,
            )
            silu_and_mul(down_input, gateup_output)
            del gateup_output

            # FP8 requantize for down GEMM
            if is_deep_gemm_e8m0_used():
                down_input_fp8, down_input_scale = sgl_per_token_group_quant_fp8(
                    down_input,
                    group_size=self.BLOCK_SIZE,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                )
            else:
                down_input_fp8, down_input_scale = trt_fp8_quantize_128(down_input, False)
            del down_input

            # Down GEMM with psum layout
            down_output = torch.empty(
                (total_tokens, K), device=device, dtype=torch.bfloat16,
            )
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (down_input_fp8, down_input_scale),
                self.w2_weight_fp8,
                down_output,
                psum,
                use_psum_layout=True,
            )
            del down_input_fp8, down_input_scale

            # Apply routing weights
            if payload.expert_topk_weights is not None:
                topk_weights = payload.expert_topk_weights
                if topk_weights.dim() == 1:
                    topk_weights = topk_weights.unsqueeze(-1)
                down_output = (down_output * topk_weights).to(torch.bfloat16)

            return CombineForwardPayload(fused_expert_output=down_output)
