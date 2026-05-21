"""DeepEPv2 ElasticBuffer-backed unified router (2D Contiguous only).

See ``集成方案/RTP集成DeepEPv2方案设计.md`` §B for the design rationale —
this router replaces both ``DeepEpLowLatencyRouter`` and
``DeepepNormalRouter`` when ``USE_DEEPEP_ELASTIC=1``. Only the
``(do_expand=True, do_cpu_sync=True)`` 2D Contiguous ``[ΣN_e, hidden]``
layout is supported, feeding the contiguous executors (TritonFusedMoe /
DeepGemmHybrid / CutlassExperts* / TrtllmFp4).

The 3D Batched (``do_cpu_sync=False``) path was removed: DeepEPv2 leaves
``num_recv_tokens_per_expert_list`` empty in that mode and ``recv_x``
stays compact 2D at offset 0, so the planned zero-copy
``.view(E_local, M_max, hidden)`` reshape would read uninitialised memory
for experts 1..E_local-1. See ``集成方案/BLOCKERS.md``. ``do_expand=False``
is similarly fail-closed (rows are deduplicated by DeepEP, which no RTP
executor can consume).
"""

import os
from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.distributed.deepep_wrapper import (
    DeepEPMode,
    DeepEPWrapper,
    DeepepWrapperConfig,
)
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    scaled_fp8_per_token_quant,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128


def _is_elastic_enabled() -> bool:
    return int(os.environ.get("USE_DEEPEP_ELASTIC", "0")) == 1


class DeepEpElasticRouter(FusedMoeDataRouter):
    """Unified DeepEPv2 ElasticBuffer dispatch/combine router.

    Only ``do_expand=True``/``do_cpu_sync=True`` (2D Contiguous,
    ``[ΣN_e, hidden]``) is supported — a drop-in for the contiguous
    executors (``DeepGemmHybridExecutor`` / ``TritonFusedMoeExecutor`` /
    ``CutlassExperts*`` / ``TrtllmFp4Executor``). Other env combinations
    are fail-closed in ``__init__``.
    """

    @classmethod
    def router_type(cls) -> RouterType:
        return RouterType.DEEPEP_ELASTIC

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Elastic router is opt-in via the USE_DEEPEP_ELASTIC env var.

        When the env var is 0 (the default), the strategy fails this gate
        and the registry falls back to the legacy LL / Normal routers.
        """
        resolver = MoeConfigResolver()
        checker.check(_is_elastic_enabled())
        checker.check(get_sm()[0] >= 9)
        checker.check(resolver.is_ep_enabled(config))
        checker.check(DeepEPWrapper.supported())
        try:
            from deep_ep import ElasticBuffer  # noqa: F401
        except ImportError:
            checker.check(False)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(config, quant_config)

        self._num_experts: int = config.expert_num
        self._num_topk: int = config.moe_k
        self._ep_size: int = config.ep_size
        self._ep_rank: int = config.ep_rank
        self._tp_size: int = config.tp_size
        self._tp_rank: int = config.tp_rank
        assert (
            self._num_experts % self._ep_size == 0
        ), f"expert_num={self._num_experts} not divisible by ep_size={self._ep_size}"
        self._expert_per_rank: int = self._num_experts // self._ep_size
        self._rank_expert_offset: int = self._ep_rank * self._expert_per_rank

        # Quantization: FP8 dispatch path + DeepGemm-friendly expert alignment.
        self._use_fp8_dispatch: bool = (
            quant_config.is_quantized
            and quant_config.quant_dtype == torch.float8_e4m3fn
        )
        self._expert_alignment: int = 128 if quant_config.is_block_quantized else 1

        # Layout is exclusively driven by DEEPEP_ELASTIC_DO_EXPAND and
        # DEEPEP_ELASTIC_DO_CPU_SYNC (parsed inside DeepepWrapperConfig);
        # USE_DEEPEP_LOW_LATENCY no longer participates. The wrapper config
        # is built first so we can read the resolved flags from a single
        # source of truth.
        if bool(config.moe_config.use_deepep_low_latency):
            ll_num_max_token_per_rank = (
                DeepepWrapperConfig.calc_low_latency_max_token_per_rank(
                    config.ll_num_max_token,
                    config.tp_size,
                    config.quant_config,
                )
            )
            deepep_config = DeepepWrapperConfig.from_config_adapter(
                self.config, ll_num_max_token_per_rank
            )
        else:
            deepep_config = DeepepWrapperConfig.from_config_adapter(self.config)

        self._do_expand: bool = deepep_config.elastic_do_expand
        self._do_cpu_sync: bool = deepep_config.elastic_do_cpu_sync
        # Strategy layer already fail-closes on do_expand=False / do_cpu_sync=False
        # (only the 2D Contiguous variants are registered). These asserts are
        # defense-in-depth against direct instantiation with invalid env
        # combinations.
        assert self._do_expand, (
            "DeepEpElasticRouter requires DEEPEP_ELASTIC_DO_EXPAND=1; "
            "do_expand=False returns deduplicated rows that no RTP "
            "executor can consume."
        )
        assert self._do_cpu_sync, (
            "DEEPEP_ELASTIC_DO_CPU_SYNC=0 is unsupported in the current "
            "integration: DeepEPv2 leaves num_recv_tokens_per_expert_list "
            "empty in that mode (psum_num_recv_tokens_per_expert is the "
            "only ground truth) and recv_x stays compact 2D at offset 0, "
            "so the planned 3D .view(E_local, M_max, hidden) reshape would "
            "read uninitialised memory for experts 1..E_local-1. See "
            "deepep integration BLOCKERS.md."
        )

        wrapper = DeepEPWrapper.get_instance(deepep_config)
        assert wrapper.mode == DeepEPMode.ELASTIC, (
            f"DeepEpElasticRouter expects DeepEPMode.ELASTIC, got {wrapper.mode}. "
            "Make sure USE_DEEPEP_ELASTIC=1 is exported before initialising the "
            "wrapper singleton."
        )
        self._buffer = wrapper.elastic_buffer

        # The EPHandle is consumed by combine() — must be cleared after each
        # finalize() so a leaked handle does not silently bind to the wrong
        # micro-batch on the next prepare().
        self._handle: Optional[Any] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def handle(self) -> Optional[Any]:
        return self._handle

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _tp_slice(
        self,
        a1: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-tp-rank slice (mirrors DeepEpLowLatencyRouter._prepare_pre_tp_slice)."""
        topk_ids = topk_ids.to(torch.int64)
        tp_size = self._tp_size
        tp_rank = self._tp_rank
        token_num = a1.size(0)
        tp_token_size = (token_num + tp_size - 1) // tp_size
        slice_begin = min(tp_token_size * tp_rank, token_num)
        slice_size = min(token_num - slice_begin, tp_token_size)
        return (
            torch.narrow(a1, 0, slice_begin, slice_size),
            torch.narrow(topk_ids, 0, slice_begin, slice_size),
            torch.narrow(topk_weights, 0, slice_begin, slice_size),
        )

    def _do_quant_fp8(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FP8 on-line quant. Mirrors DeepepNormalRouterBase._do_quant."""
        if self.quant_config.is_block_quantized:
            if is_deep_gemm_e8m0_used():
                return sgl_per_token_group_quant_fp8(
                    a1,
                    128,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                )
            return trt_fp8_quantize_128(a1, False)
        if self.quant_config.is_per_act_token:
            a1_q, a1_scale = scaled_fp8_per_token_quant(a1, None)
            assert a1_q.shape[1] % 128 == 0
            a1_scale = a1_scale.repeat(1, a1_q.shape[1] // 128)
            return a1_q, a1_scale
        raise ValueError(
            f"Unsupported FP8 quant config for elastic dispatch: {self.quant_config}"
        )

    def _finalize_post_tp_gather(
        self,
        combined_x: torch.Tensor,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """All-gather across TP ranks, dropping right-padding (same as LL/Normal)."""
        assert combined_x.dim() == 2
        assert extra_finalize_args is not None
        assert "original_num_tokens" in extra_finalize_args
        tp_size = self._tp_size
        original_num_tokens: int = extra_finalize_args["original_num_tokens"]
        tp_token_size = (original_num_tokens + tp_size - 1) // tp_size
        if tp_size > 1:
            if combined_x.size(0) < tp_token_size:
                padding = torch.empty(
                    size=(tp_token_size - combined_x.size(0), combined_x.size(1)),
                    device=combined_x.device,
                    dtype=combined_x.dtype,
                )
                combined_x = torch.cat([combined_x, padding], dim=0)
            gathered = all_gather(combined_x, group=Group.TP).reshape(
                tp_size * tp_token_size, -1
            )
            combined_x = gathered[:original_num_tokens, :]
        return combined_x

    # ------------------------------------------------------------------
    # prepare (= dispatch)
    # ------------------------------------------------------------------
    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        assert self._handle is None, "elastic EPHandle leaked from previous step"
        if a1_scale is not None or a2_scale is not None:
            raise ValueError(
                "DeepEpElasticRouter handles fp8 quantization internally; "
                "external a1_scale / a2_scale must be None."
            )

        act_dtype = a1.dtype

        tp_a1, tp_topk_ids, tp_topk_weights = self._tp_slice(
            a1, topk_ids, topk_weights
        )

        if self._use_fp8_dispatch:
            assert (
                self.quant_config.is_block_quantized
                or self.quant_config.is_per_act_token
            ), (
                "ElasticRouter FP8 dispatch requires block_quantized or "
                f"per_act_token quant config, got {self.quant_config}"
            )
            x_payload = self._do_quant_fp8(tp_a1)
        else:
            x_payload = tp_a1

        recv_x, recv_topk_idx, recv_topk_weights, handle, event = (
            self._buffer.dispatch(
                x=x_payload,
                topk_idx=tp_topk_ids,
                topk_weights=tp_topk_weights,
                num_experts=self._num_experts,
                expert_alignment=self._expert_alignment,
                do_expand=self._do_expand,
                do_cpu_sync=self._do_cpu_sync,
                # Use async-with-compute-stream so dispatch returns a real
                # CUDA event we can wait on. async_with_compute_stream=False
                # ran dispatch on a separate comm stream without capturing
                # an event, leaving downstream reads racing against in-
                # flight writes — see iter 3 anomaly trap diagnosis.
                async_with_compute_stream=True,
            )
        )
        # ESSENTIAL: ElasticBuffer.dispatch may complete asynchronously on
        # its internal comm stream; do_cpu_sync only synchronises the CPU
        # side (per-expert counts), not the GPU data tensors. Without this
        # wait, downstream executor reads recv_x while it's still being
        # written, producing garbage that cascades into NaN a few layers
        # later. Matches vLLM PR #41183 prepare_finalize/deepep_v2.py.
        # NB: EventOverlap wraps an inner CUDA event; that inner event is
        # None when async_with_compute_stream=False, so we have to peek
        # inside before calling current_stream_wait (which asserts inner
        # event is not None).
        if event is not None and getattr(event, "event", None) is not None:
            event.current_stream_wait()
        self._handle = handle

        if isinstance(recv_x, tuple):
            expert_x, expert_x_scale = recv_x
        else:
            expert_x, expert_x_scale = recv_x, None

        num_per_expert = handle.num_recv_tokens_per_expert_list
        # ElasticBuffer returns an empty list when 0 tokens were dispatched
        # to this rank (init warmup or skewed traffic). Downstream
        # executors (DeepGemmMasked / CutlassBatched*) assert
        # `len(expert_num_tokens) == E_local`, so pad with zeros up to the
        # expected length here so the contract holds uniformly.
        if len(num_per_expert) == 0:
            num_per_expert = [0] * self._expert_per_rank
        elif len(num_per_expert) != self._expert_per_rank:
            raise AssertionError(
                f"ElasticBuffer handle.num_recv_tokens_per_expert_list len "
                f"{len(num_per_expert)} differs from E_local "
                f"{self._expert_per_rank}; ep_size={self._ep_size}"
            )
        expert_num_tokens = torch.tensor(
            num_per_expert,
            device=expert_x.device,
            dtype=torch.int32,
        )

        # 2D Contiguous (`do_expand=True, do_cpu_sync=True`) — DeepEPv2
        # unrolls the topk dimension into rows: every received row is a
        # single (orig_token, expert) pair. Shapes empirically observed:
        #   recv_x.shape         = (N_recv, hidden)  (or (fp8, sf) tuple)
        #   recv_topk_idx        = None
        #   recv_topk_weights    = (N_recv,)   (1D, one weight per slot)
        #   num_recv_per_expert  = list[E_local]  (cumsum gives slot ranges)
        # Synthesize per-row `expert_topk_ids` (global id, shape
        # `[N_recv, 1]`) from the per-expert count and reshape weights
        # to `[N_recv, 1]` so the contiguous executors see a
        # self-consistent `num_topk=1` layout.
        if recv_topk_idx is None:
            offsets = []
            for local_eid, cnt in enumerate(num_per_expert):
                offsets.extend(
                    [self._rank_expert_offset + local_eid] * int(cnt)
                )
            if offsets:
                expert_topk_ids = torch.tensor(
                    offsets, device=expert_x.device, dtype=torch.int64
                ).unsqueeze(1)
            else:
                expert_topk_ids = torch.empty(
                    (0, 1), device=expert_x.device, dtype=torch.int64
                )
        else:
            expert_topk_ids = torch.where(
                recv_topk_idx == -1,
                self._num_experts - 1 if self._rank_expert_offset == 0 else 0,
                recv_topk_idx + self._rank_expert_offset,
            )

        if recv_topk_weights is not None and recv_topk_weights.dim() == 1:
            recv_topk_weights = recv_topk_weights.unsqueeze(1)

        meta = ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=num_per_expert,
        )

        return ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_scale=expert_x_scale,
            expert_x_origin_dtype=act_dtype,
            expert_topk_ids=expert_topk_ids,
            expert_topk_weights=recv_topk_weights,
            expert_tokens_meta=meta,
        )

    # ------------------------------------------------------------------
    # finalize (= combine)
    # ------------------------------------------------------------------
    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
        skip_allreduce: bool = False,
    ) -> torch.Tensor:
        assert (
            self._handle is not None
        ), "DeepEpElasticRouter.finalize() called without a live EPHandle"

        x = payload.fused_expert_output
        assert (
            x.dtype == torch.bfloat16
        ), f"ElasticBuffer.combine requires bfloat16 input, got {x.dtype}"

        # With do_expand=True (always, asserted in __init__), each dispatched
        # row already carries its own (token, expert) identity — weights are
        # baked into the per-expert layout and combine performs a simple
        # gather, not a topk reduction, so topk_weights=None.
        combined_x, _, combine_event = self._buffer.combine(
            x=x,
            handle=self._handle,
            topk_weights=None,
            async_with_compute_stream=True,
        )
        if (
            combine_event is not None
            and getattr(combine_event, "event", None) is not None
        ):
            combine_event.current_stream_wait()
        self._handle = None

        combined_x = self._finalize_post_tp_gather(combined_x, extra_finalize_args)
        return combined_x
