"""FlashInfer FP8 groupwise SM100 executor for MoE per-block FP8.

Uses FlashInfer's `group_gemm_fp8_nt_groupwise` kernel on SM100,
which supports 2SM MMA for large prefill batches. This provides
higher prefill throughput than DeepGEMM's 1SM contiguous path.

Weight scales are recomputed from the stored FP8 data at init time
(the FP8 weights have already been re-quantized via requant_weight_ue8m0
during model loading). The float32 scales are stored alongside the
UE8M0-packed scales used by DeepGEMM executors.
"""

import logging
import math
from typing import Any, Dict, Optional

import torch

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
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
)
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.models_py.utils.math import align, ceil_div
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)

BLOCK_SIZE = 128


def _recompute_float32_scales(weight_fp8: torch.Tensor) -> torch.Tensor:
    """Recompute per-block float32 scales from stored FP8 weights.

    The weights have already been re-quantized with UE8M0 ceil-rounded scales
    via requant_weight_ue8m0() during model loading. We recompute the same
    scales here: amax / 448.0, then ceil to power-of-2 (UE8M0 rounding).

    Args:
        weight_fp8: [E, N, K] or [N, K] fp8_e4m3fn tensor

    Returns:
        scales: [E, N//128, K//128] or [N//128, K//128] float32 tensor
    """
    has_batch = weight_fp8.dim() == 3
    if has_batch:
        E, N, K = weight_fp8.shape
        w_flat = weight_fp8.reshape(-1, K)
    else:
        N, K = weight_fp8.shape
        w_flat = weight_fp8

    N_total = w_flat.shape[0]
    N_padded = align(N_total, BLOCK_SIZE)
    K_padded = align(K, BLOCK_SIZE)

    # Pad to block-aligned dimensions
    w_padded = torch.zeros(
        (N_padded, K_padded), dtype=w_flat.dtype, device=w_flat.device
    )
    w_padded[:N_total, :K] = w_flat

    # Compute per-block amax
    w_view = w_padded.view(N_padded // BLOCK_SIZE, BLOCK_SIZE, K_padded // BLOCK_SIZE, BLOCK_SIZE)
    w_amax = w_view.abs().float().amax(dim=(1, 3)).clamp(min=1e-4)

    # Scale = amax / 448.0, rounded up to power-of-2 (UE8M0)
    sf = w_amax / 448.0
    sf = torch.pow(2.0, torch.ceil(torch.log2(sf.abs())))

    if has_batch:
        sf = sf.view(E, ceil_div(N, BLOCK_SIZE), ceil_div(K, BLOCK_SIZE))

    return sf


def _build_m_indptr(expert_offsets: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Build FlashInfer m_indptr from expert cumulative offsets.

    FlashInfer requires m_indptr values to be padded to multiples of 4.

    Args:
        expert_offsets: [E+1] int32 tensor with cumulative token offsets
        num_experts: number of experts

    Returns:
        m_indptr: [E+1] int32 tensor, padded to multiples of 4
    """
    m_indptr = expert_offsets[:num_experts + 1].clone()
    # Pad each cumulative value to multiple of 4
    # We only pad the segment sizes, keeping cumulative structure
    padded = torch.zeros_like(m_indptr)
    padded[0] = 0
    for i in range(num_experts):
        seg_size = m_indptr[i + 1] - m_indptr[i]
        padded_size = ((seg_size + 3) // 4) * 4
        padded[i + 1] = padded[i] + padded_size
    return padded


class FlashInferFp8GroupwiseExecutor(FusedMoeExpertExecutor):
    """FlashInfer FP8 groupwise SM100 executor.

    Uses group_gemm_fp8_nt_groupwise with:
    - scale_granularity_mnk=(1, 128, 128)
    - scale_major_mode="MN"
    - mma_sm=2 for large M (prefill), mma_sm=1 for small M (decode)
    """

    BLOCK_SIZE = 128
    EXPERT_ALIGNMENT = 128
    # Threshold: use 2SM when avg tokens per expert >= this
    MMA_2SM_THRESHOLD = 256

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.FLASHINFER_FP8_GROUPWISE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(resolver.is_bf16(config))
        checker.check(get_sm()[0] >= 10)  # SM100+ only

        # Check FlashInfer groupwise kernel availability
        try:
            from flashinfer.gemm import group_gemm_fp8_nt_groupwise  # noqa: F401
            checker.check(True)
        except ImportError:
            checker.check(False)

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
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.top_k = config.moe_k

        # FP8 weights [E, N, K]
        self.w13_weight = weights[W.moe_w1]
        self.w2_weight = weights[W.moe_w2]

        self.E, self.N, self.K = self.w13_weight.shape
        assert self.N % 2 == 0

        # Recompute float32 per-block scales from stored FP8 data.
        # Shape: [E, N//128, K//128] → transpose to [E, K//128, N//128] for MN-major
        w13_scale = _recompute_float32_scales(self.w13_weight)
        w2_scale = _recompute_float32_scales(self.w2_weight)

        # FlashInfer MN-major b_scale: (batch_size, k // block_size, n // block_size)
        self.w13_scale_mn = w13_scale.permute(0, 2, 1).contiguous()
        self.w2_scale_mn = w2_scale.permute(0, 2, 1).contiguous()

        logger.info(
            f"FlashInferFp8GroupwiseExecutor: E={self.E}, N={self.N}, K={self.K}, "
            f"w13_scale_mn={self.w13_scale_mn.shape}, w2_scale_mn={self.w2_scale_mn.shape}"
        )

    @property
    def local_num_experts(self) -> int:
        return self.w13_weight.size(0)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        from flashinfer.gemm import group_gemm_fp8_nt_groupwise

        assert payload.expert_x is not None
        assert payload.expert_x_scale is not None
        assert payload.expert_topk_ids is not None
        assert payload.expert_topk_weights is not None
        assert payload.expert_tokens_meta is not None

        hidden_states = payload.expert_x  # [M, K] bf16
        topk_idx = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights

        if payload.expert_tokens_meta.expert_num_tokens_cpu is not None:
            num_tokens_per_expert = payload.expert_tokens_meta.expert_num_tokens_cpu
        elif payload.expert_tokens_meta.expert_num_tokens is not None:
            num_tokens_per_expert = payload.expert_tokens_meta.expert_num_tokens.cpu().tolist()
        else:
            raise ValueError("expert_tokens_meta must have expert_num_tokens")

        if isinstance(num_tokens_per_expert, torch.Tensor):
            num_tokens_per_expert = num_tokens_per_expert.tolist()

        E = self.num_experts_per_partition
        N = self.N
        K = self.K
        device = hidden_states.device

        # Pad tokens per expert to multiples of 4 (FlashInfer requirement)
        padded_tokens = [((t + 3) // 4) * 4 for t in num_tokens_per_expert]
        total_padded = sum(padded_tokens)

        if total_padded <= 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    hidden_states.shape, device=device, dtype=torch.bfloat16
                ),
            )

        # Build m_indptr for FlashInfer
        m_indptr = torch.zeros(E + 1, dtype=torch.int32, device=device)
        for i in range(E):
            m_indptr[i + 1] = m_indptr[i] + padded_tokens[i]

        # Scatter tokens to per-expert contiguous layout and quantize to FP8
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        # Scatter: reorder tokens by expert assignment
        num_tokens_gpu = torch.tensor(
            padded_tokens, dtype=torch.int32, device=device
        )
        expert_start_loc = torch.zeros_like(num_tokens_gpu)

        input_fp8 = torch.empty(
            (total_padded, K), device=device, dtype=torch.float8_e4m3fn
        )
        # Activation scale: (K//128, total_padded) for MN-major
        input_scale = torch.zeros(
            (K // BLOCK_SIZE, total_padded), device=device, dtype=torch.float32
        )

        output_index = torch.empty_like(topk_idx)
        m_indices = torch.empty(
            total_padded, device=device, dtype=torch.int32
        )

        ep_scatter(
            hidden_states,
            payload.expert_x_scale,
            topk_idx,
            num_tokens_gpu,
            expert_start_loc,
            input_fp8,
            input_scale,
            m_indices,
            output_index,
            scale_ue8m0=False,
        )

        # Quantize scattered activation to FP8 with per-group float32 scales
        # Actually, the scatter already handles FP8 input.
        # For FlashInfer, we need activation scales in MN-major: (K//128, cum_m)
        # The scatter output scale is already in the right format if scale_ue8m0=False

        # Determine mma_sm based on average tokens per expert
        avg_tokens = total_padded // max(E, 1)
        mma_sm = 2 if avg_tokens >= self.MMA_2SM_THRESHOLD else 1

        # === FC1: Gate+Up GEMM ===
        # A=[total_padded, K] @ B=[E, 2N_inter, K]^T → C=[total_padded, 2N_inter]
        fc1_output = group_gemm_fp8_nt_groupwise(
            a=input_fp8,
            b=self.w13_weight,
            a_scale=input_scale,
            b_scale=self.w13_scale_mn,
            m_indptr=m_indptr,
            scale_granularity_mnk=(1, BLOCK_SIZE, BLOCK_SIZE),
            scale_major_mode="MN",
            mma_sm=mma_sm,
            out_dtype=torch.bfloat16,
        )

        dispose_tensor(input_fp8)

        # === Activation: SiLU(gate) * up ===
        fc1_act = torch.empty(
            (total_padded, N // 2), device=device, dtype=torch.bfloat16
        )
        silu_and_mul(fc1_act, fc1_output)
        dispose_tensor(fc1_output)

        # === Quantize FC1 output to FP8 for FC2 ===
        fc2_input_fp8, fc2_input_scale = sgl_per_token_group_quant_fp8(
            fc1_act,
            group_size=BLOCK_SIZE,
            column_major_scales=True,
            scale_tma_aligned=False,
            scale_ue8m0=False,
        )
        dispose_tensor(fc1_act)

        # fc2_input_scale shape after column_major: (N_inter//128, total_padded)
        # This is already MN-major format for FlashInfer a_scale

        # Build m_indptr for FC2 (same expert assignment, same padding)
        # === FC2: Down GEMM ===
        # A=[total_padded, N_inter] @ B=[E, K, N_inter]^T → C=[total_padded, K]
        fc2_output = group_gemm_fp8_nt_groupwise(
            a=fc2_input_fp8,
            b=self.w2_weight,
            a_scale=fc2_input_scale,
            b_scale=self.w2_scale_mn,
            m_indptr=m_indptr,
            scale_granularity_mnk=(1, BLOCK_SIZE, BLOCK_SIZE),
            scale_major_mode="MN",
            mma_sm=mma_sm,
            out_dtype=torch.bfloat16,
        )

        dispose_tensor(fc2_input_fp8)
        dispose_tensor(fc2_input_scale)

        # === Gather: scatter-reduce back to original token order ===
        gather_out = torch.empty(
            hidden_states.shape, device=device, dtype=torch.bfloat16
        )
        ep_gather(fc2_output, topk_idx, topk_weights, output_index, gather_out)

        return CombineForwardPayload(fused_expert_output=gather_out)
