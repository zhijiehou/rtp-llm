import os
from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.distributed.deepep_wrapper import (
    DeepEPMode,
    DeepEPWrapper,
    DeepepWrapperConfig,
    is_elastic_buffer,
)
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
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
from rtp_llm.models_py.utils.arch import get_sm

# DeepEP kernels quantize dispatch inputs in 128 element chunks.
DEEPEP_QUANT_BLOCK_SIZE = 128
# DeepEP Low-Latency supports hidden sizes
SUPPORTED_HIDDEN_SIZES = [1536, 2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]


class DeepEpLowLatencyRouter(FusedMoeDataRouter):
    """
    A data router for Mixture-of-Experts that utilizes deep_ep's low-latency communication primitives.

    This router dispatches tokens to experts and receives results from experts across all ep ranks.
    """

    @classmethod
    def router_type(cls):
        return RouterType.DEEPEP_LOW_LATENCY

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepEpLowLatencyRouter can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(get_sm()[0] >= 9)
        checker.check(resolver.is_ep_enabled(config))
        checker.check(resolver.use_low_latency(config))
        checker.check(DeepEPWrapper.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)

        # Determine use_fp8_dispatch based on quant_config
        use_fp8_dispatch = (
            quant_config.is_quantized
            and quant_config.quant_dtype == torch.float8_e4m3fn
        )

        # DeepEpLowLatency-specific initialization
        self._num_experts = config.expert_num
        self._ll_num_max_token_per_rank = (
            DeepepWrapperConfig.calc_low_latency_max_token_per_rank(
                config.ll_num_max_token, config.tp_size, config.quant_config
            )
        )
        deepep_config = DeepepWrapperConfig.from_config_adapter(
            self.config, self._ll_num_max_token_per_rank
        )
        wrapper = DeepEPWrapper.get_instance(deepep_config)
        assert (
            wrapper.mode == DeepEPMode.LOW_LATENCY
        ), "DeepEP mode should be LOW_LATENCY"
        self._buffer = wrapper.buffer
        # Cache buffer kind once: ElasticBuffer (v2 unified) vs legacy.Buffer
        # (v1 low_latency_dispatch / low_latency_combine). The two have wholly
        # different signatures and return tuples; downstream prepare/finalize
        # branch on this flag instead of doing isinstance checks every call.
        self._is_elastic = is_elastic_buffer(self._buffer)
        self._num_topk = wrapper.num_topk
        self._num_max_dispatch_tokens_per_rank = wrapper.ll_num_max_token_per_rank
        self._use_fp8_dispatch = use_fp8_dispatch
        self._zero_copy = False
        self._async_finish = False
        self._return_recv_hook = False
        self._opt_level = int(os.environ.get("ACCL_LOW_LATENCY_OPTIMIZE", 1))
        # legacy.Buffer.low_latency_dispatch returns a Tuple-shaped handle;
        # ElasticBuffer.dispatch returns an `EPHandle` instance. Either way the
        # router only uses it as an opaque token to feed into the matching
        # combine call site, so we widen the annotation to `Any`.
        self._handle: Any = None
        self._use_accl_ep = wrapper.use_accl_ep

    @property
    def handle(self) -> Any:
        return self._handle

    def _prepare_pre_tp_slice(
        self,
        a1: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice the dispatch activation, topk_ids and topk_weights by tp.
        Args:
            a1 (torch.Tensor): The dispatch activation tensor.
            topk_ids (torch.Tensor): The topk ids tensor.
            topk_weights (torch.Tensor): The topk weights tensor.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The sliced dispatch activation, topk_ids and topk_weights.
        """
        # Convert topk_ids to int64
        topk_ids = topk_ids.to(torch.int64)
        # Slice by tp
        tp_size = self.config.tp_size
        tp_rank = self.config.tp_rank
        token_num = a1.size(0)
        tp_token_size = (token_num + tp_size - 1) // tp_size
        slice_begin = min(tp_token_size * tp_rank, token_num)
        slice_size = min(token_num - slice_begin, tp_token_size)
        tp_dispatch_input = torch.narrow(a1, 0, slice_begin, slice_size)
        tp_topk_ids = torch.narrow(topk_ids, 0, slice_begin, slice_size)
        tp_topk_weights = torch.narrow(topk_weights, 0, slice_begin, slice_size)

        return tp_dispatch_input, tp_topk_ids, tp_topk_weights

    def _normal_prepare(
        self, dispatch_args: dict[str, Any], tp_topk_weights: torch.Tensor
    ):
        """Normal prepare for DeepEP Low-Latency.
        Args:
            dispatch_args (dict[str, Any]): Arguments for dispatching tokens to experts.
            tp_topk_weights (torch.Tensor): Topk weights tensor for this tp rank.
        """
        # Calculate expected_m
        tp_num_tokens = dispatch_args["x"].size(0)
        expected_m = max(
            1,
            int(
                tp_num_tokens
                * self.config.ep_size
                * self._num_topk
                // self._num_experts
            ),
        )

        # Dispatch tokens
        expert_x, expert_num_tokens, self._handle, _, _ = (
            self._buffer.low_latency_dispatch(**dispatch_args)
        )
        if self._use_fp8_dispatch:
            assert isinstance(expert_x, tuple), "expert_x should be a tuple"
            expert_x, expert_x_scale = expert_x[0], expert_x[1]
        else:
            assert isinstance(expert_x, torch.Tensor), "expert_x should be a tensor"
            expert_x = expert_x
            expert_x_scale = None

        # Return expert forward payload
        return ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_scale=expert_x_scale,
            expert_x_origin_dtype=dispatch_args["x"].dtype,
            expert_topk_ids=dispatch_args["topk_idx"],
            expert_topk_weights=tp_topk_weights,
            expert_tokens_meta=ExpertTokensMetadata(
                expected_m=expected_m,
                expert_num_tokens=expert_num_tokens,
            ),
        )

    def _elastic_prepare(
        self,
        tp_dispatch_input: torch.Tensor,
        tp_topk_ids: torch.Tensor,
        tp_topk_weights: torch.Tensor,
    ) -> ExpertForwardPayload:
        """Elastic-backend dispatch path for low-latency routing.

        PR #605 unifies low-latency and high-throughput EP under a single
        `ElasticBuffer.dispatch`, so this collapses the legacy two-path setup
        (`low_latency_dispatch` + `low_latency_combine`) to the same calls used
        by the normal router. The handle returned here is `EPHandle`, not the
        v1 tuple; `finalize` branches on `self._is_elastic` to handle that.
        """

        tp_num_tokens = tp_dispatch_input.size(0)
        expected_m = max(
            1,
            int(
                tp_num_tokens
                * self.config.ep_size
                * self._num_topk
                // self._num_experts
            ),
        )

        (
            output,
            recv_topk_idx,
            recv_topk_weights,
            self._handle,
            _,
        ) = self._buffer.dispatch(
            x=tp_dispatch_input,
            topk_idx=tp_topk_ids,
            topk_weights=tp_topk_weights,
            num_experts=self._num_experts,
            num_max_tokens_per_rank=self._num_max_dispatch_tokens_per_rank,
        )

        if self._use_fp8_dispatch:
            assert isinstance(output, tuple), "expert_x should be a tuple in fp8 mode"
            expert_x, expert_x_scale = output
        else:
            assert isinstance(output, torch.Tensor)
            expert_x, expert_x_scale = output, None

        expert_num_tokens = torch.tensor(
            self._handle.num_recv_tokens_per_expert_list,
            device=expert_x.device,
            dtype=torch.int32,
        )

        return ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_scale=expert_x_scale,
            expert_x_origin_dtype=tp_dispatch_input.dtype,
            expert_topk_ids=recv_topk_idx if recv_topk_idx is not None else tp_topk_ids,
            expert_topk_weights=recv_topk_weights
            if recv_topk_weights is not None
            else tp_topk_weights,
            expert_tokens_meta=ExpertTokensMetadata(
                expected_m=expected_m,
                expert_num_tokens=expert_num_tokens,
            ),
        )

    def _elastic_finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Elastic-backend combine path. ElasticBuffer.combine signature is
        (x, handle, topk_weights, bias, num_sms, num_qps, ...) returning
        (combined_x, combined_topk_weights, event)."""

        assert self._handle is not None
        combined_x, _, _ = self._buffer.combine(
            x=payload.fused_expert_output,
            handle=self._handle,
            topk_weights=topk_weights,
        )
        return combined_x

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        """
        Dispatches tokens to experts across all ep ranks.
        """
        # Check payload data
        num_tokens, hidden_size = a1.size()
        assert (
            hidden_size in SUPPORTED_HIDDEN_SIZES
            and hidden_size % DEEPEP_QUANT_BLOCK_SIZE == 0
        )
        tp_num_tokens = (num_tokens + self.config.tp_size - 1) // self.config.tp_size
        assert (
            tp_num_tokens <= self._num_max_dispatch_tokens_per_rank
        ), f"tp_num_tokens {tp_num_tokens} > self._num_max_dispatch_tokens_per_rank {self._num_max_dispatch_tokens_per_rank}"
        assert topk_ids.size(0) == num_tokens and topk_weights.size(0) == num_tokens
        assert (
            topk_ids.size(1) == self._num_topk
            and topk_weights.size(1) == self._num_topk
        )
        # Check quantization
        if self._use_fp8_dispatch:
            assert self.quant_config.is_block_quantized or (
                self.quant_config.is_per_act_token and self._use_accl_ep
            ), "DeepEP Low-Latency only supports fp8 block quantization or per_act_token quantization with ACCL-EP"
        elif self.quant_config.is_per_group_fp4:
            pass
        else:
            assert not self.quant_config.is_quantized
        # Check handle
        assert self._handle is None

        # Slice dispatch activation, topk_ids and topk_weights by tp
        tp_dispatch_input, tp_topk_ids, tp_topk_weights = self._prepare_pre_tp_slice(
            a1, topk_ids, topk_weights
        )

        if self._is_elastic:
            # Elastic backend: v2 unified dispatch (no separate low_latency
            # variant). Quant flags (`use_fp8`, `round_scale`, `pertoken_quant`)
            # are absorbed by ElasticBuffer's `use_fp8_dispatch` constructor
            # arg and tensor dtype, so we do not pass them through here.
            return self._elastic_prepare(
                tp_dispatch_input, tp_topk_ids, tp_topk_weights
            )

        # Prepare dispatch basic arguments
        dispatch_args = {
            "x": tp_dispatch_input,
            "topk_idx": tp_topk_ids,
            "num_max_dispatch_tokens_per_rank": self._num_max_dispatch_tokens_per_rank,
            "num_experts": self._num_experts,
            "use_fp8": self._use_fp8_dispatch,
            "async_finish": self._async_finish,
            "return_recv_hook": self._return_recv_hook,
        }
        # Set quantization config for DeepEP low latency dispatch
        if self.quant_config.is_block_quantized and is_deep_gemm_e8m0_used():
            dispatch_args.update({"round_scale": True, "use_ue8m0": True})
        elif self.quant_config.is_per_act_token:
            dispatch_args.update({"pertoken_quant": True})

        # Normal prepare
        expert_payload = self._normal_prepare(dispatch_args, tp_topk_weights)

        return expert_payload

    def _normal_finalize(self, combine_args: dict[str, Any]) -> torch.Tensor:
        """Normal finalize for DeepEP Low-Latency.
        Args:
            combine_args (dict[str, Any]): Arguments for combining expert outputs.
        """
        # Normal finalize
        combined_x, _, _ = self._buffer.low_latency_combine(**combine_args)

        return combined_x

    def _finalize_post_tp_gather(
        self, combined_x: torch.Tensor, extra_finalize_args: Optional[Dict[str, Any]]
    ) -> torch.Tensor:
        """Finalize post tp gather for DeepEP Low-Latency.
        Args:
            combined_x (torch.Tensor): Combined output from all tp ranks.
            extra_finalize_args (Optional[Dict[str, Any]]): Extra finalize arguments.
        """
        # Check input data
        assert combined_x.dim() == 2
        assert extra_finalize_args is not None
        assert "original_num_tokens" in extra_finalize_args
        # Get original number of tokens
        tp_size = self.config.tp_size
        original_num_tokens = extra_finalize_args["original_num_tokens"]
        tp_token_size = (original_num_tokens + tp_size - 1) // tp_size
        if tp_size > 1:
            # combine_x.size(0) might be 0
            if combined_x.size(0) < tp_token_size:
                # Pad combined output if needed
                padding_combined_x = torch.empty(
                    size=(tp_token_size - combined_x.size(0), combined_x.size(1)),
                    device=combined_x.device,
                    dtype=combined_x.dtype,
                )
                combined_x = torch.cat([combined_x, padding_combined_x], dim=0)
            # Gather combined output from all tp ranks
            gatherd_output = all_gather(combined_x, group=Group.TP).reshape(
                tp_size * tp_token_size, -1
            )
            combined_x = gatherd_output[:original_num_tokens, :]
        return combined_x

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
        skip_allreduce: bool = False,
    ) -> torch.Tensor:
        """
        Combines expert outputs back to all original ranks.
        Weight application and reduction happens in the combine kernel.
        """
        # Check handle
        assert (
            self._handle is not None
        ), "DeepEP Low-latency combine handle is missing for finalize()."

        # Convert topk_ids to int64
        topk_ids = topk_ids.to(torch.int64)

        if self._is_elastic:
            combined_x = self._elastic_finalize(payload, topk_weights)
        else:
            # Prepare combine basic arguments
            combine_args: Dict[str, Any] = {
                "x": payload.fused_expert_output,
                "topk_idx": topk_ids,
                "topk_weights": topk_weights,
                "handle": self._handle,
                "zero_copy": self._zero_copy,
                "async_finish": self._async_finish,
                "return_recv_hook": self._return_recv_hook,
            }
            if self._use_accl_ep:
                combine_args.update({"opt_level": self._opt_level})

            # Normal finalize
            combined_x = self._normal_finalize(combine_args)

        # Finalize post tp gather
        combined_x = self._finalize_post_tp_gather(combined_x, extra_finalize_args)
        # reset handle
        self._handle = None

        return combined_x
