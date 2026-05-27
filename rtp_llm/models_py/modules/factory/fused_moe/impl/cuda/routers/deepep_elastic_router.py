"""DeepEPv2 ElasticBuffer-backed unified router (prefill + decode cudagraph).

See ``集成方案/RTP集成DeepEPv2方案设计.md`` §B for the design rationale —
this router replaces both ``DeepEpLowLatencyRouter`` and
``DeepepNormalRouter`` when ``USE_DEEPEP_ELASTIC=1``.

Two ElasticBuffer dispatch modes are now supported:

* **Prefill / default** — ``(do_expand=True, do_cpu_sync=True)`` → tight 2D
  Contiguous ``[ΣN_e, hidden]`` layout, expert-grouped, feeds
  ``CutlassExperts*`` / ``DeepGemmHybridExecutor`` / ``TritonFusedMoe`` /
  ``TrtllmFp4Executor`` (the "contiguous" executor family).
* **Decode cudagraph** — ``(do_expand=False, do_cpu_sync=False)`` →
  ``[worst_case_N, hidden]`` in original token order with per-row
  ``recv_topk_idx`` (``-1`` marks non-local / padding slots).  No D2H /
  ``cudaStreamSynchronize`` happens on the GPU side, so the path is
  CUDA Graph capture-friendly.  Mirrors vLLM ``DeepEPV2PrepareAndFinalize``
  decode mode (PR #41183).

Other ``(do_expand, do_cpu_sync)`` combinations stay fail-closed: mixed
modes are not exercised by any DeepEPv2 caller we know of.  ``do_expand=
False`` with ``do_cpu_sync=True`` returns deduplicated rows that no RTP
executor can consume; ``do_expand=True`` with ``do_cpu_sync=False`` leaves
``num_recv_tokens_per_expert_list`` empty while still expecting per-expert
slicing.
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

    Two layouts are supported, selected by env at construction time
    (defaults stay on prefill so legacy callers see no behavioural change):

    * ``DEEPEP_ELASTIC_DO_EXPAND=1, DEEPEP_ELASTIC_DO_CPU_SYNC=1`` (default)
      → 2D Contiguous ``[ΣN_e, hidden]``, expert-grouped, drop-in for
      ``DeepGemmHybridExecutor`` / ``TritonFusedMoeExecutor`` /
      ``CutlassExperts*`` / ``TrtllmFp4Executor``.
    * ``DEEPEP_ELASTIC_DO_EXPAND=0, DEEPEP_ELASTIC_DO_CPU_SYNC=0``
      → vLLM-style decode cudagraph: ``[worst_case_N, hidden]`` in
      original token order, per-row ``recv_topk_idx`` with ``-1`` for
      non-local / padding slots, ``expert_tokens_meta=None``.  Pairs with
      executors that natively guard ``topk_idx == -1`` inside their GPU
      kernels (``CutlassExpertsFp8`` / ``CutlassExpertsW4a8Int4PerChannel``
      / ``DeepGemmMaskedExecutorV2``).

    Mixed combinations (``(True, False)`` / ``(False, True)``) remain
    fail-closed in ``__init__`` — DeepEPv2 has no caller for them and
    silently produces inconsistent metadata in either direction.
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
        self._use_fp4: bool = (
            MoeConfigResolver().get_quant_method(config) == "modelopt_fp4"
        )
        self._use_local_expert_ids: bool = (
            self._use_fp8_dispatch or self._use_fp4
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
        # Strategy layer fail-closes on mismatched combinations (only the
        # ``*EpElasticContiguousStrategy`` and ``*EpElasticDecodeStrategy``
        # subclasses are registered). This assert is defense-in-depth
        # against direct instantiation with invalid env combinations.
        # Two layouts are supported:
        #   (True,  True)  → 2D Contiguous prefill (default)
        #   (False, False) → vLLM-style decode cudagraph
        # Mixed combinations have no caller and would produce inconsistent
        # metadata (e.g. empty num_recv_tokens_per_expert_list with
        # expand=True, deduplicated rows with cpu_sync=True).
        assert (self._do_expand and self._do_cpu_sync) or (
            (not self._do_expand) and (not self._do_cpu_sync)
        ), (
            "DeepEpElasticRouter supports two layouts: prefill "
            "(DEEPEP_ELASTIC_DO_EXPAND=1, DEEPEP_ELASTIC_DO_CPU_SYNC=1) and "
            "decode cudagraph (DEEPEP_ELASTIC_DO_EXPAND=0, "
            "DEEPEP_ELASTIC_DO_CPU_SYNC=0). Got do_expand="
            f"{self._do_expand}, do_cpu_sync={self._do_cpu_sync}. Mixed "
            "combinations are fail-closed because DeepEPv2 produces "
            "inconsistent metadata in those modes."
        )
        # True iff the (False, False) decode cudagraph path is in effect.
        self._use_decode_cudagraph: bool = not self._do_cpu_sync

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
        if a1.size(0) == 0:
            hidden = a1.size(1)
            a1_q = torch.empty(
                (0, hidden), dtype=torch.float8_e4m3fn, device=a1.device
            )
            scale_cols = hidden // 128 if hidden >= 128 else 1
            a1_scale = torch.empty(
                (0, scale_cols), dtype=torch.float32, device=a1.device
            )
            return a1_q, a1_scale
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

        # NaN-input guard (collective-safe, GPU-side sanitisation).
        #
        # Background:
        # qwen35_moe (and a few other DeepEP MoE models) produces NaN at
        # layer 1 on some rank during engine init warmup (numerical edge
        # case with the synthetic init input). Legacy DeepEP tolerates
        # this silently. DeepEPv2 dispatch has
        # `ptx::deduplicate(dst_expert_idx, lane_idx)` which asserts when
        # input is all-NaN → gate(NaN) → argmax(NaN) → 0 → topk slots
        # all-equal-zero, lane-level duplicates trip the assertion.
        #
        # Cuda graph friendliness:
        # The previous implementation did `.any().item()` + Python
        # `if tainted:` branch, which triggers a host sync and is
        # forbidden inside `torch.cuda.graph()` capture. Rewritten in
        # vLLM-style "unconditional `torch.where`" form (mirrors
        # `vllm/v1/attention/ops/dcp_alltoall.py:65-87`):
        # everything is GPU elementwise, no `.item() / .cpu()`, no
        # Python branch, capturable. NaN-free inputs see only a few
        # extra GPU elementwise nops; cost is negligible.
        #
        # Dispatch is a collective op, so we must apply the same
        # sanitisation on every rank uniformly — done implicitly because
        # this code runs on every rank with the same logic.
        # Default on; disable with =0 to expose the underlying NaN for
        # model-side debugging (will lose cuda-graph capture).
        nan_guard_active = int(os.environ.get("DEEPEP_ELASTIC_NAN_GUARD", "1"))
        if nan_guard_active:
            # (1) hidden states: NaN → 0
            a1 = torch.where(torch.isnan(a1), torch.zeros_like(a1), a1)

            # (2) topk_weights: NaN → 1/k uniform
            inv_k = 1.0 / float(self._num_topk)
            topk_weights = torch.where(
                torch.isnan(topk_weights),
                torch.full_like(topk_weights, inv_k),
                topk_weights,
            )

            # (3) topk_ids: DeepEPv2 dispatch's `ptx::deduplicate` fails
            # when **any two slots in a row are equal** (not just when
            # the whole row collapses to 0). NaN-tainted gate produces
            # the most extreme case (all slots == 0) but capture warmup
            # with zero hidden states can also produce mid-degree
            # collisions (a few slots equal). Detect "row has any
            # duplicate slot" GPU-side via sort + neighbor compare, and
            # replace such rows with a round-robin id sequence so every
            # slot within a row is distinct.
            #
            # GPU-only detection: torch.sort + neighbor equality + .any
            # all return GPU tensors. No `.item() / .cpu()`. The
            # `if topk_ids.size(1) > 1` guard reads tensor *metadata*
            # (host const at trace time, folded by graph capture into
            # a single branch), not data, so it's safe inside
            # torch.cuda.graph().
            if topk_ids.size(1) > 1:
                rows = topk_ids.size(0)
                k_topk = topk_ids.size(1)
                rr = (
                    torch.arange(
                        rows * k_topk,
                        device=topk_ids.device,
                        dtype=topk_ids.dtype,
                    )
                    % self._num_experts
                ).reshape(rows, k_topk)
                sorted_ids, _ = torch.sort(topk_ids, dim=-1)
                has_dup = (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any(
                    dim=-1, keepdim=True
                )
                topk_ids = torch.where(has_dup, rr, topk_ids)

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
            if (
                expert_x_scale is not None
                and self.quant_config.is_per_act_token
                and expert_x_scale.dim() == 2
                and expert_x_scale.size(1) > 1
            ):
                expert_x_scale = expert_x_scale[:, 0].contiguous()
        else:
            expert_x, expert_x_scale = recv_x, None

        if self._use_decode_cudagraph:
            # vLLM-style (False, False) decode cudagraph path. ElasticBuffer
            # returns:
            #   recv_x: [worst_case_N, hidden] (or (fp8, sf) tuple) in
            #     ORIGINAL token order — not expert-grouped, may include
            #     non-local rows and worst-case padding whose contents are
            #     untouched
            #   recv_topk_idx: [worst_case_N, num_topk] of LOCAL expert IDs
            #     with -1 marking non-local / padding slots
            #   recv_topk_weights: [worst_case_N, num_topk] aligned with
            #     recv_topk_idx
            #   handle.num_recv_tokens_per_expert_list: empty list (push
            #     happens only when do_cpu_sync=True, see
            #     DeepEP/csrc/elastic/buffer.hpp:1003-1008)
            #
            # Two cudagraph invariants must be preserved here:
            #   1. NO ``.item() / .cpu() / .tolist()`` — would break capture
            #   2. ``expert_tokens_meta=None`` — passing it with a non-None
            #      ``expert_num_tokens_cpu`` would trigger
            #      ``cutlass_moe.py:128-131`` D2H even though the downstream
            #      kernel never needs it (cf. vLLM ``deepep_v2.py``
            #      docstring "Expert kernel sorts internally").
            assert recv_topk_idx is not None, (
                "(False, False) dispatch must return per-row recv_topk_idx, "
                "got None — ElasticBuffer contract violation."
            )
            if self._use_local_expert_ids:
                expert_topk_ids = recv_topk_idx
            else:
                expert_topk_ids = torch.where(
                    recv_topk_idx == -1,
                    self._num_experts - 1 if self._rank_expert_offset == 0 else 0,
                    recv_topk_idx + self._rank_expert_offset,
                )
            return ExpertForwardPayload(
                expert_x=expert_x,
                expert_x_scale=expert_x_scale,
                expert_x_origin_dtype=act_dtype,
                expert_topk_ids=expert_topk_ids,
                expert_topk_weights=recv_topk_weights,
                expert_tokens_meta=None,
            )

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
        # Synthesize per-row `expert_topk_ids` (local or global id, shape
        # `[N_recv, 1]`) from the per-expert count and reshape weights
        # to `[N_recv, 1]` so the contiguous executors see a
        # self-consistent `num_topk=1` layout.
        if recv_topk_idx is None:
            eid_offset = 0 if self._use_local_expert_ids else self._rank_expert_offset
            offsets = []
            for local_eid, cnt in enumerate(num_per_expert):
                offsets.extend([eid_offset + local_eid] * int(cnt))
            if offsets:
                expert_topk_ids = torch.tensor(
                    offsets, device=expert_x.device, dtype=torch.int64
                ).unsqueeze(1)
            else:
                expert_topk_ids = torch.empty(
                    (0, 1), device=expert_x.device, dtype=torch.int64
                )
        else:
            if self._use_local_expert_ids:
                expert_topk_ids = recv_topk_idx
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
