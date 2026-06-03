# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py
# but make some modifications for RTP-LLM
# Licensed under the Apache License, Version 2.0
import logging
import math
from typing import Any, Dict, Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    is_deep_gemm_e8m0_used,
    m_grouped_fp8_gemm_nt_contiguous,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_and_mul,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    tma_align_input_scale,
)
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128
from rtp_llm.utils.model_weight import W


@triton.jit
def _build_m_indices_kernel(
    tokens_per_expert_ptr,
    m_indices_ptr,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)
    offset = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens = tl.load(tokens_per_expert_ptr + offset, mask=offset < num_experts, other=0)
    cumsum = tl.cumsum(tokens) - tokens
    start = tl.sum(tl.where(offset < cur_expert, tokens, 0))
    count = tl.load(tokens_per_expert_ptr + cur_expert)
    base = m_indices_ptr + start
    off = tl.arange(0, BLOCK_E)
    for s in tl.range(0, count, BLOCK_E, num_stages=4):
        tl.store(base + s + off, cur_expert)


def build_m_indices_triton(
    num_recv_tokens_per_expert: torch.Tensor,
    all_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    num_experts = num_recv_tokens_per_expert.shape[0]
    m_indices = torch.empty(all_tokens, device=device, dtype=torch.int32)
    if all_tokens > 0:
        _build_m_indices_kernel[(num_experts,)](
            num_recv_tokens_per_expert,
            m_indices,
            num_experts=num_experts,
            BLOCK_E=128,
            BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
        )
    return m_indices


def align_up_math(n: int, alignment: int = 128) -> int:
    return int(math.ceil(n / alignment)) * alignment


class DeepGemmContiguousExecutor(FusedMoeExpertExecutor):
    BLOCK_SIZE = 128
    EXPERT_ALIGNMENT = 128
    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.DEEPGEMM_CONTINUOUS

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepGemmContiguousExecutor can handle the configuration"""
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
        checker.check(not config.enable_cuda_graph)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.num_experts = config.expert_num

        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1

        self.top_k = config.moe_k
        self.activation = config.activation_type
        self.renormalize = True
        self.use_fp8_w8a8 = True
        self.use_block_quant = True

        # 权重初始化
        self.w13_weight = weights[W.moe_w1]
        self.w2_weight = weights[W.moe_w2]
        self.w13_weight_scale_inv = weights[W.moe_s1]
        self.w2_weight_scale_inv = weights[W.moe_s2]
        self.w13_weight_scale = None
        self.w2_weight_scale = None

        self.E, self.N, self.K = self.w13_weight.size()
        assert self.N % 2 == 0
        assert self.w2_weight.size(0) == self.E
        assert self.w2_weight.size(1) == self.K
        assert self.w2_weight.size(2) == self.N // 2

        self.w13_weight_fp8 = (
            self.w13_weight,
            self.w13_weight_scale_inv,
        )
        self.w2_weight_fp8 = (
            self.w2_weight,
            self.w2_weight_scale_inv,
        )

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "hidden_states_fp8 is not initialized"
        assert (
            payload.expert_x_scale is not None
        ), "hidden_states_scale is not initialized"
        assert (
            payload.expert_topk_weights is not None
        ), "expert_topk_weights is not initialized"
        assert (
            payload.expert_tokens_meta is not None
        ), "expert_tokens_meta is not initialized"

        hidden_states_fp8 = payload.expert_x
        hidden_states_scale = payload.expert_x_scale
        topk_weights = payload.expert_topk_weights

        num_recv_tokens_per_expert = payload.expert_tokens_meta.expert_num_tokens_cpu

        num_recv_tokens_per_expert = [
            align_up_math(x, self.EXPERT_ALIGNMENT) for x in num_recv_tokens_per_expert
        ]
        all_tokens: int = sum(num_recv_tokens_per_expert)

        if all_tokens <= 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    hidden_states_fp8.shape,
                    device=hidden_states_fp8.device,
                    dtype=torch.bfloat16,
                ),
            )

        _, K = hidden_states_fp8.size()
        N = self.w13_weight.size(1)
        hidden_states_fp8_device = hidden_states_fp8.device

        # 数据已由 DeepEP dispatch (do_expand=true) 按专家连续排列，
        # 无需 ep_scatter，只需生成 m_indices 告诉 grouped GEMM 每行属于哪个专家
        tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert, device=hidden_states_fp8_device, dtype=torch.int32,
        )
        m_indices = build_m_indices_triton(tokens_per_expert_gpu, all_tokens, hidden_states_fp8_device)

        # 等待 dispatch 通信完成（m_indices 构建与 dispatch 通信尾部重叠）
        dispatch_event = payload.dispatch_event
        if dispatch_event is not None:
            dispatch_event.current_stream_wait()

        # GEMM1: gate + up projection
        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        input_scale = hidden_states_scale
        if not is_deep_gemm_e8m0_used():
            input_scale = tma_align_input_scale(input_scale)
        m_grouped_fp8_gemm_nt_contiguous(
            (hidden_states_fp8, input_scale),
            self.w13_weight_fp8,
            gateup_output,
            m_indices,
            disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
        )
        dispose_tensor(hidden_states_fp8)

        # Fused SiLU + FP8 quantize: single-pass over gateup_output
        gateup_output = gateup_output.view(-1, N)
        down_input_fp8, down_input_scale = sgl_per_token_group_quant_fp8(
            gateup_output,
            group_size=self.BLOCK_SIZE,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=is_deep_gemm_e8m0_used(),
            fuse_silu_and_mul=True,
        )
        del gateup_output
        if not is_deep_gemm_e8m0_used():
            down_input_scale = tma_align_input_scale(down_input_scale)
        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )

        # GEMM2: down projection
        m_grouped_fp8_gemm_nt_contiguous(
            (down_input_fp8, down_input_scale),
            self.w2_weight_fp8,
            down_output,
            m_indices,
            disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
        )
        del down_input_fp8, down_input_scale

        # expand 模式下每行是独立的 token-expert pair，
        # 直接乘以路由权重即可，无需 ep_gather 重排
        if topk_weights is not None:
            down_output *= topk_weights

        return CombineForwardPayload(fused_expert_output=down_output)
