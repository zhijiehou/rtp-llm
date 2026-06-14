import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, final

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import (
    ExecutorType,
    RouterType,
)


@dataclass
class ExpertTokensMetadata:
    """
    Metadata regarding expert-token routing.
    """

    expected_m: Optional[int] = None
    expert_num_tokens: Optional[torch.Tensor] = None
    expert_num_tokens_cpu: Optional[Union[List[int], torch.Tensor]] = None


@dataclass
class ExpertForwardPayload:
    """
    Represents the data payload dispatched to experts for computation.
    """

    expert_x: torch.Tensor
    expert_x_origin_dtype: Optional[torch.dtype] = None
    expert_x_scale: Optional[torch.Tensor] = None
    expert_tokens_meta: Optional[ExpertTokensMetadata] = None
    expert_topk_ids: Optional[torch.Tensor] = None
    expert_topk_weights: Optional[torch.Tensor] = None
    expert_ids_are_local: bool = False


@dataclass
class CombineForwardPayload:
    """
    Represents the data payload for combining the expert outputs.
    """

    fused_expert_output: torch.Tensor


class FusedMoeDataRouter(ABC):
    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        """Initialize FusedMoeDataRouter with standard parameters.

        Args:
            config: MOE configuration adapter
            quant_config: Quantization configuration
        """
        self.config = config
        self.quant_config = quant_config

    @classmethod
    def router_type(cls) -> RouterType:
        raise NotImplementedError

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if this router can handle the given configuration.

        Subclasses should override this method to check router-specific conditions.

        Args:
            checker: ConditionChecker instance from MoeStrategy
            config: Model initialization parameters
        """
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        raise NotImplementedError


class FusedMoeExpertExecutor(ABC):
    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """Initialize FusedMoeExpertExecutor with standard parameters.

        Args:
            config: MOE configuration adapter
            quant_config: Quantization configuration
            weights: Model weights dictionary
        """
        self.config = config
        self.quant_config = quant_config
        self.weights = weights

    @classmethod
    def executor_type(cls) -> ExecutorType:
        raise NotImplementedError

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if this executor can handle the given configuration.

        Subclasses should override this method to check executor-specific conditions.

        Args:
            checker: ConditionChecker instance from MoeStrategy
            config: Model initialization parameters
        """
        pass

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int64

    @abstractmethod
    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        raise NotImplementedError


@final
class FusedMoe(torch.nn.Module):
    _instance_counter = 0

    def __init__(
        self,
        router: FusedMoeDataRouter,
        fused_experts: FusedMoeExpertExecutor,
        expert_num: int,
    ):
        super().__init__()
        self.router = router
        self.fused_experts = fused_experts
        self.expert_num = expert_num
        self._profile_step = 0
        # Assign a unique layer index to distinguish multiple MoE layers
        FusedMoe._instance_counter += 1
        self._layer_id = FusedMoe._instance_counter
        # warmup: skip first N inference requests before recording
        self._profile_warmup = int(os.environ.get("PROFILE_FUSED_MOE_WARMUP", "3"))
        # active: number of consecutive inference requests to record
        self._profile_active = int(os.environ.get("PROFILE_FUSED_MOE_ACTIVE", "5"))
        # torch.profiler instance, created lazily when recording starts
        self._profiler = None

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return self.fused_experts.topk_ids_dtype

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        expert_map: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        extra_expert_args: Optional[Dict[str, Any]] = None,
        extra_finalize_args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:

        a1 = hidden_states
        nvtx_enabled = os.environ.get("PROFILE_FUSED_MOE", "0") == "1"
        layer_tag = f"MoE_L{self._layer_id}"

        # ---- dispatch (router.prepare = EP all2all dispatch) ----
        if nvtx_enabled:
            torch.cuda.nvtx.range_push(
                f"{layer_tag}/dispatch_tokens={hidden_states.shape[0]}"
            )
        expert_payload = self.router.prepare(
            a1,
            a1_scale,
            a2_scale,
            topk_weights,
            topk_ids,
        )
        if nvtx_enabled:
            torch.cuda.nvtx.range_pop()

        if expert_payload.expert_topk_ids is None:
            expert_payload.expert_topk_ids = topk_ids
        if expert_payload.expert_topk_weights is None:
            expert_payload.expert_topk_weights = topk_weights

        # ---- expert compute (GEMM) ----
        if nvtx_enabled:
            torch.cuda.nvtx.range_push(f"{layer_tag}/gemm")
        if expert_payload.expert_x.numel() == 0:
            # This happens when none of the tokens from the all2all reach this
            # EP rank. Also, note that this is only relevant for CUDAGraph
            # incompatible all2all kernels like the DeepEP high-throughput
            # kernels. CUDAGraph compatible all2all kernels like the pplx
            # kernels and the DeepEP low-latency kernels are always batched
            # and can never run into the tensor.numel() == 0 case.
            combine_payload = CombineForwardPayload(
                fused_expert_output=torch.empty_like(
                    expert_payload.expert_x, dtype=a1.dtype
                )
            )
        else:
            combine_payload = self.fused_experts.execute(
                expert_payload,
                activation=activation,
                expert_map=expert_map,
                a2_scale=a2_scale,
                apply_router_weight_on_input=apply_router_weight_on_input,
                extra_expert_args=extra_expert_args,
            )
        if nvtx_enabled:
            torch.cuda.nvtx.range_pop()

        # pass a1.shape to finalize for shape check
        if extra_finalize_args is None:
            extra_finalize_args = {"a1_shape": a1.shape}
        else:
            extra_finalize_args.update({"a1_shape": a1.shape})

        extra_finalize_args.update({"original_num_tokens": hidden_states.size(0)})

        # ---- combine (router.finalize = EP all2all combine) ----
        if nvtx_enabled:
            torch.cuda.nvtx.range_push(f"{layer_tag}/combine")
        output = self.router.finalize(
            combine_payload,
            expert_payload.expert_topk_weights,
            expert_payload.expert_topk_ids,
            apply_router_weight_on_input,
            extra_finalize_args,
        )
        if nvtx_enabled:
            torch.cuda.nvtx.range_pop()

        assert (
            output.shape == hidden_states.shape
        ), f"output batch size mismatch: expected {hidden_states.shape}, got {output.shape}"

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        expert_map: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        extra_expert_args: Optional[Dict[str, Any]] = None,
        extra_finalize_args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        profile_enabled = os.environ.get("PROFILE_FUSED_MOE", "0") == "1"
        # Only profile specific layers to avoid Kineto singleton conflict
        # when multiple MoE layers run concurrently.
        # PROFILE_NUM_LAYERS controls how many layers (starting from layer 1) to profile.
        profile_num_layers = int(os.environ.get("PROFILE_NUM_LAYERS", "1"))
        # Skip forwards with too few tokens (e.g. fake DP dispatch, warmup padding).
        # PROFILE_MIN_TOKENS=0 disables the filter.
        profile_min_tokens = int(os.environ.get("PROFILE_MIN_TOKENS", "16"))
        num_tokens = hidden_states.shape[0]
        token_count_ok = (profile_min_tokens == 0) or (num_tokens >= profile_min_tokens)
        should_profile_this_layer = (
            profile_enabled
            and (self._layer_id <= profile_num_layers)
            and token_count_ok
        )

        if should_profile_this_layer:
            self._profile_step += 1
            in_active = (
                self._profile_warmup
                < self._profile_step
                <= self._profile_warmup + self._profile_active
            )
            just_finished = (
                self._profile_step == self._profile_warmup + self._profile_active + 1
            )

            if in_active and self._profiler is None:
                try:
                    from torch.profiler import ProfilerActivity, profile

                    self._profiler = profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                    )
                    self._profiler.__enter__()
                except Exception as profiler_ex:
                    print(
                        f"[FusedMoe profiler] layer={self._layer_id} failed to start: {profiler_ex}"
                    )
                    self._profiler = None

            output = self._forward_impl(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=inplace,
                activation=activation,
                expert_map=expert_map,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                apply_router_weight_on_input=apply_router_weight_on_input,
                extra_expert_args=extra_expert_args,
                extra_finalize_args=extra_finalize_args,
            )

            if in_active and self._profiler is not None:
                try:
                    self._profiler.step()
                except Exception:
                    pass

            if just_finished and self._profiler is not None:
                try:
                    torch.cuda.synchronize()
                    self._profiler.__exit__(None, None, None)
                    rank = 0
                    if (
                        torch.distributed.is_available()
                        and torch.distributed.is_initialized()
                    ):
                        rank = torch.distributed.get_rank()
                    default_path = (
                        f"/root/hzj/fused_moe_layer{self._layer_id}_rank{rank}.json"
                    )
                    trace_path = os.environ.get(
                        "PROFILE_FUSED_MOE_OUTPUT", default_path
                    )
                    self._profiler.export_chrome_trace(trace_path)
                    print(
                        f"[FusedMoe profiler] layer={self._layer_id} rank={rank} "
                        f"active={self._profile_active} steps, timeline saved to {trace_path}"
                    )
                except Exception as export_ex:
                    print(
                        f"[FusedMoe profiler] layer={self._layer_id} export failed: {export_ex}"
                    )
                finally:
                    self._profiler = None

            return output

        return self._forward_impl(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            activation=activation,
            expert_map=expert_map,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=extra_expert_args,
            extra_finalize_args=extra_finalize_args,
        )
