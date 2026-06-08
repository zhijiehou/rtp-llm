import logging
import os
import time
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
class _MoeProfiler:
    """MoE layer profiler using CUDA events for GPU timing, gated by PROFILE_FUSED_MOE=1.

    Measures actual GPU execution time for prepare/execute/finalize via CUDA events,
    plus wall-clock span from first to last MoE layer (with single GPU sync at end).
    """

    enabled: bool = False
    _initialized: bool = False
    _layer: int = 0
    _req: int = 0
    _prep_events: list = []
    _exec_events: list = []
    _fin_events: list = []
    _span_start: float = 0
    _num_layers: int = 61
    _logger = None

    @classmethod
    def init(cls) -> None:
        if cls._initialized:
            return
        cls.enabled = os.environ.get("PROFILE_FUSED_MOE", "0") == "1"
        cls._num_layers = int(os.environ.get("PROFILE_NUM_LAYERS", "61"))
        if cls.enabled:
            cls._logger = logging.getLogger("moe_profiler")
        cls._initialized = True

    @classmethod
    def on_forward_start(cls) -> torch.cuda.Event:
        if cls._layer == 0:
            cls._span_start = time.perf_counter()
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    @classmethod
    def on_phase_boundary(cls) -> torch.cuda.Event:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    @classmethod
    def on_forward_end(
        cls, ev_start: torch.cuda.Event, ev_prep: torch.cuda.Event, ev_exec: torch.cuda.Event
    ) -> None:
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_end.record()
        cls._prep_events.append((ev_start, ev_prep))
        cls._exec_events.append((ev_prep, ev_exec))
        cls._fin_events.append((ev_exec, ev_end))
        cls._layer += 1

        if cls._layer >= cls._num_layers:
            torch.cuda.synchronize()
            span = (time.perf_counter() - cls._span_start) * 1000
            cls._req += 1
            n = cls._num_layers
            ps = sum(s.elapsed_time(e) for s, e in cls._prep_events[-n:])
            es = sum(s.elapsed_time(e) for s, e in cls._exec_events[-n:])
            fs = sum(s.elapsed_time(e) for s, e in cls._fin_events[-n:])
            gpu_total = ps + es + fs
            cls._logger.warning(  # type: ignore[union-attr]
                f"[MOE-PROFILE req#{cls._req}] "
                f"span={span:.1f}ms gpu_total={gpu_total:.1f}ms | "
                f"prepare={ps:.1f}ms({ps/gpu_total*100:.0f}%) "
                f"execute={es:.1f}ms({es/gpu_total*100:.0f}%) "
                f"finalize={fs:.1f}ms({fs/gpu_total*100:.0f}%) | "
                f"per_layer: prep={ps/n:.3f} exec={es/n:.3f} fin={fs/n:.3f}ms"
            )
            cls._layer = 0
            cls._prep_events.clear()
            cls._exec_events.clear()
            cls._fin_events.clear()


class FusedMoe(torch.nn.Module):
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

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return self.fused_experts.topk_ids_dtype

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
        _MoeProfiler.init()
        _profiling = _MoeProfiler.enabled and hidden_states.size(0) > 10

        a1 = hidden_states

        if _profiling:
            _ev0 = _MoeProfiler.on_forward_start()
        expert_payload = self.router.prepare(
            a1,
            a1_scale,
            a2_scale,
            topk_weights,
            topk_ids,
        )

        if expert_payload.expert_topk_ids is None:
            expert_payload.expert_topk_ids = topk_ids
        if expert_payload.expert_topk_weights is None:
            expert_payload.expert_topk_weights = topk_weights

        if _profiling:
            _ev1 = _MoeProfiler.on_phase_boundary()
        if expert_payload.expert_x.numel() == 0:
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

        if _profiling:
            _ev2 = _MoeProfiler.on_phase_boundary()
        # pass a1.shape to finalize for shape check
        if extra_finalize_args is None:
            extra_finalize_args = {"a1_shape": a1.shape}
        else:
            extra_finalize_args.update({"a1_shape": a1.shape})

        extra_finalize_args.update({"original_num_tokens": hidden_states.size(0)})

        output = self.router.finalize(
            combine_payload,
            expert_payload.expert_topk_weights,
            expert_payload.expert_topk_ids,
            apply_router_weight_on_input,
            extra_finalize_args,
        )

        if _profiling:
            _MoeProfiler.on_forward_end(_ev0, _ev1, _ev2)

        assert (
            output.shape == hidden_states.shape
        ), f"output batch size mismatch: expected {hidden_states.shape}, got {output.shape}"

        return output
