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

        # NaN-input guard (collective-safe, in-place sanitisation).
        # Iter 9 diagnosis: the model produces NaN at layer 1 on
        # rank 1 during engine init warmup (qwen35_moe has a known
        # numerical edge case with the synthetic init input). Legacy
        # DeepEP tolerates this silently — its dispatch doesn't
        # assert on topk_idx duplicates. Elastic DeepEPv2 dispatch
        # has `ptx::deduplicate(dst_expert_idx, lane_idx)` which
        # fails when input is all-NaN → gate(NaN) → argmax(NaN) → 0
        # → topk=[0,0,...,0] all duplicates.
        #
        # Mitigation: replace NaN-tainted inputs IN PLACE before
        # dispatch — substitute zeros for `a1`, uniform 1/k for
        # `topk_weights`, and a distinct round-robin id sequence
        # for `topk_ids`. Dispatch is a collective operation, so we
        # cannot early-return on a subset of ranks (that deadlocks
        # the comm); we must call it on every rank with safe inputs.
        # Default on; disable with =0 to expose the underlying NaN
        # for model-side debugging.
        nan_guard_active = int(os.environ.get("DEEPEP_ELASTIC_NAN_GUARD", "1"))
        if nan_guard_active:
            tainted = bool(
                torch.isnan(a1).any().item()
                or torch.isnan(topk_weights).any().item()
            )
            if tainted:
                if not getattr(DeepEpElasticRouter, "_logged_nan_guard", False):
                    print(
                        f"[DeepEpElasticRouter] NAN_GUARD fired (rank "
                        f"{self._ep_rank}) — sanitising NaN-tainted input "
                        f"before dispatch (a1 → zeros, topk → "
                        f"round-robin, weights → 1/k). Mirrors legacy "
                        f"DeepEP silent-tolerance.",
                        flush=True,
                    )
                    DeepEpElasticRouter._logged_nan_guard = True
                a1 = torch.zeros_like(a1)
                # Build distinct ids per row so ptx::deduplicate stays
                # happy: row r, slot k → expert (r*num_topk + k) %
                # num_experts. With num_topk <= num_experts (always
                # true) the per-row k slots are guaranteed distinct.
                rows = topk_ids.size(0)
                k = topk_ids.size(1)
                base = (
                    torch.arange(rows, device=topk_ids.device).unsqueeze(1) * k
                    + torch.arange(k, device=topk_ids.device).unsqueeze(0)
                )
                topk_ids = (base % self._num_experts).to(topk_ids.dtype)
                topk_weights = torch.full_like(topk_weights, 1.0 / float(k))

        # FULL_BYPASS: skip both dispatch and combine entirely. Pretend the
        # MoE produced zeros. Use to confirm whether the elastic kernel
        # calls themselves (rather than their return values) are corrupting
        # other ranks' memory via P2P side effects.
        if int(os.environ.get("DEEPEP_ELASTIC_FULL_BYPASS", "0")):
            if not getattr(DeepEpElasticRouter, "_logged_full_bypass", False):
                print(
                    "[DeepEpElasticRouter] FULL_BYPASS enabled — skipping "
                    "dispatch and combine entirely, returning zero payload",
                    flush=True,
                )
                DeepEpElasticRouter._logged_full_bypass = True
            self._handle = "__FULL_BYPASS__"
            zero_x = torch.zeros((1, a1.size(1)), dtype=torch.bfloat16, device=a1.device)
            zero_ids = torch.zeros((1, 1), dtype=torch.int64, device=a1.device)
            zero_w = torch.zeros((1, 1), dtype=topk_weights.dtype, device=a1.device)
            return ExpertForwardPayload(
                expert_x=zero_x,
                expert_x_scale=None,
                expert_x_origin_dtype=act_dtype,
                expert_topk_ids=zero_ids,
                expert_topk_weights=zero_w,
                expert_tokens_meta=ExpertTokensMetadata(
                    expert_num_tokens=torch.zeros(
                        self._expert_per_rank, dtype=torch.int32, device=a1.device
                    ),
                    expert_num_tokens_cpu=[0] * self._expert_per_rank,
                ),
            )

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

        # Handle metadata diagnostic — dump per-rank what combine will use
        # for reverse-routing. Used in iter 8 to disprove a suspected
        # combine reverse-mapping bug (turned out 3 ranks just had
        # identical input topk_idx so identical output was correct).
        if int(os.environ.get("DEEPEP_ELASTIC_DUMP_HANDLE", "0")):
            n_done = getattr(DeepEpElasticRouter, "_handle_dump_done", 0)
            max_n = int(os.environ.get("DEEPEP_ELASTIC_DUMP_HANDLE_MAX", "4"))
            if n_done < max_n:
                def _summ(t, name):
                    if t is None:
                        return f"{name}=None"
                    if not hasattr(t, "shape"):
                        return f"{name}=type:{type(t).__name__} val={t!r}"
                    flat = t.detach().to(torch.int64).flatten() if t.numel() else t
                    head = flat[: min(8, flat.numel())].tolist() if flat.numel() else []
                    s = float(flat.sum().item()) if flat.numel() else 0.0
                    return (
                        f"{name}.shape={tuple(t.shape)} dtype={t.dtype} "
                        f"sum={s} head={head}"
                    )
                attrs = (
                    "num_recv_tokens_per_expert_list",
                    "psum_num_recv_tokens_per_scaleup_rank",
                    "psum_num_recv_tokens_per_expert",
                    "recv_src_metadata",
                    "dst_buffer_slot_idx",
                    "topk_idx",
                    "num_recv_tokens",
                    "num_experts",
                    "expert_alignment",
                    "num_max_tokens_per_rank",
                    "do_expand",
                )
                lines = [f"  rank={self._ep_rank}:"]
                for a in attrs:
                    v = getattr(handle, a, "<missing>")
                    lines.append("    " + _summ(v, a))
                print(
                    "[DeepEpElasticRouter] HANDLE_DUMP\n" + "\n".join(lines),
                    flush=True,
                )
                DeepEpElasticRouter._handle_dump_done = n_done + 1

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

        # FULL_BYPASS short-circuit: skip combine entirely, return zeros
        # shaped like the original input (so downstream residual add works).
        if self._handle == "__FULL_BYPASS__":
            self._handle = None
            original_num_tokens = (
                extra_finalize_args.get("original_num_tokens")
                if extra_finalize_args
                else None
            )
            if original_num_tokens is None:
                original_num_tokens = payload.fused_expert_output.size(0)
            hidden = payload.fused_expert_output.size(1)
            return torch.zeros(
                (original_num_tokens, hidden),
                dtype=torch.bfloat16,
                device=payload.fused_expert_output.device,
            )

        x = payload.fused_expert_output
        assert (
            x.dtype == torch.bfloat16
        ), f"ElasticBuffer.combine requires bfloat16 input, got {x.dtype}"

        # Always log first N finalize calls — print value stats (max abs,
        # mean, NaN count) of the executor output so we see whether it's
        # SILENTLY DEGRADED (e.g. all-zero, abnormally large, or off-scale
        # vs legacy) even when not NaN.
        if int(os.environ.get("DEEPEP_ELASTIC_DEBUG_ANOMALY", "1")):
            seq = getattr(DeepEpElasticRouter, "_finalize_seq", 0) + 1
            DeepEpElasticRouter._finalize_seq = seq
            max_prints = int(
                os.environ.get("DEEPEP_ELASTIC_FINALIZE_DEBUG_MAX", "6")
            )
            n_printed = getattr(DeepEpElasticRouter, "_finalize_printed", 0)
            if n_printed < max_prints and x.numel() > 0:
                with torch.no_grad():
                    xf = x.detach().to(torch.float32)
                    nan_cnt = int(torch.isnan(xf).sum().item())
                    inf_cnt = int(torch.isinf(xf).sum().item())
                    safe = xf[~torch.isnan(xf) & ~torch.isinf(xf)]
                    max_abs = float(safe.abs().max().item()) if safe.numel() else 0.0
                    mean_abs = float(safe.abs().mean().item()) if safe.numel() else 0.0
                    zero_frac = float((safe == 0).float().mean().item()) if safe.numel() else 0.0
                print(
                    f"[DeepEpElasticRouter] FINALIZE seq={seq} ep_rank={self._ep_rank} "
                    f"executor_x.shape={tuple(x.shape)} nan={nan_cnt} inf={inf_cnt} "
                    f"max_abs={max_abs:.4g} mean_abs={mean_abs:.4g} zero_frac={zero_frac:.3f}",
                    flush=True,
                )
                DeepEpElasticRouter._finalize_printed = n_printed + 1

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

        # Cross-rank combined_x diff check: gather combined_x from all 4 ranks
        # and check bit-for-bit equality. If 3 ranks see identical data, combine
        # semantics is broadcasting instead of per-rank-correct reverse-mapping.
        # Gated to first N invocations; ALL ranks must always do the gather to
        # avoid NCCL deadlock.
        if int(os.environ.get("DEEPEP_ELASTIC_CROSS_RANK_DIFF", "0")):
            n_done = getattr(DeepEpElasticRouter, "_xrank_diff_done", 0)
            max_n = int(os.environ.get("DEEPEP_ELASTIC_CROSS_RANK_DIFF_MAX", "3"))
            if n_done < max_n:
                try:
                    import torch.distributed as dist
                    gathered = [
                        torch.empty_like(combined_x) for _ in range(self._ep_size)
                    ]
                    dist.all_gather(gathered, combined_x.contiguous())
                    with torch.no_grad():
                        per_rank_diff_vs_0 = []
                        for r in range(self._ep_size):
                            if r == 0:
                                per_rank_diff_vs_0.append(0.0)
                            else:
                                d = (
                                    gathered[r].to(torch.float32)
                                    - gathered[0].to(torch.float32)
                                ).abs().max().item()
                                per_rank_diff_vs_0.append(float(d))
                        # also per-rank sum-of-abs as a fingerprint
                        rank_sums = [
                            float(g.to(torch.float32).abs().sum().item())
                            for g in gathered
                        ]
                    print(
                        f"[DeepEpElasticRouter] CROSS_RANK_DIFF n={n_done} "
                        f"ep_rank={self._ep_rank} "
                        f"shape={tuple(combined_x.shape)} "
                        f"diff_vs_rank0={per_rank_diff_vs_0} "
                        f"abs_sum_per_rank={rank_sums}",
                        flush=True,
                    )
                    DeepEpElasticRouter._xrank_diff_done = n_done + 1
                except Exception as e:
                    print(
                        f"[DeepEpElasticRouter] CROSS_RANK_DIFF skipped: {e!r}",
                        flush=True,
                    )

        # Always-log stats on combine output for first N calls, same idea.
        if int(os.environ.get("DEEPEP_ELASTIC_DEBUG_ANOMALY", "1")):
            seq2 = getattr(DeepEpElasticRouter, "_combine_seq", 0) + 1
            DeepEpElasticRouter._combine_seq = seq2
            max_p2 = int(os.environ.get("DEEPEP_ELASTIC_COMBINE_DEBUG_MAX", "6"))
            n_p2 = getattr(DeepEpElasticRouter, "_combine_printed", 0)
            if n_p2 < max_p2 and combined_x.numel() > 0:
                with torch.no_grad():
                    cf = combined_x.detach().to(torch.float32)
                    nan_cnt = int(torch.isnan(cf).sum().item())
                    inf_cnt = int(torch.isinf(cf).sum().item())
                    safe = cf[~torch.isnan(cf) & ~torch.isinf(cf)]
                    max_abs = float(safe.abs().max().item()) if safe.numel() else 0.0
                    mean_abs = float(safe.abs().mean().item()) if safe.numel() else 0.0
                    zero_frac = float((safe == 0).float().mean().item()) if safe.numel() else 0.0
                print(
                    f"[DeepEpElasticRouter] COMBINE seq={seq2} ep_rank={self._ep_rank} "
                    f"combined_x.shape={tuple(combined_x.shape)} nan={nan_cnt} inf={inf_cnt} "
                    f"max_abs={max_abs:.4g} mean_abs={mean_abs:.4g} zero_frac={zero_frac:.3f}",
                    flush=True,
                )
                DeepEpElasticRouter._combine_printed = n_p2 + 1

        combined_x = self._finalize_post_tp_gather(combined_x, extra_finalize_args)

        # Isolation experiment: replace our combined_x with zeros to see
        # whether NaN downstream comes from our values (then zeros should
        # stop the cascade) or from the model itself (then zeros wouldn't
        # help). Gated; default off.
        if int(os.environ.get("DEEPEP_ELASTIC_FORCE_ZERO_OUTPUT", "0")):
            if not getattr(DeepEpElasticRouter, "_logged_force_zero", False):
                print(
                    f"[DeepEpElasticRouter] FORCE_ZERO_OUTPUT enabled — "
                    f"returning zeros instead of combined_x shape={tuple(combined_x.shape)}",
                    flush=True,
                )
                DeepEpElasticRouter._logged_force_zero = True
            combined_x = torch.zeros_like(combined_x)

        return combined_x
