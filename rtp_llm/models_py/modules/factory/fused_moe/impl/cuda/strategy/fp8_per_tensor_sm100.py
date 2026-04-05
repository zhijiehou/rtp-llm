"""CUDA FP8 PerTensor SM100 (Blackwell) strategies.

These strategies gate on SM100 hardware and use CutlassSm100ExpertsFp8
which has higher priority than the generic CutlassExpertsFp8.
On non-SM100 hardware, these strategies will not match, and the
existing generic FP8 per-tensor strategies will be used instead.
"""

from typing import Any

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class CudaFp8PerTensorSm100NoDPStrategy(MoeStrategy):
    """SM100 FP8 PerTensor single GPU strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(
            config.moe_strategy == "fp8_per_tensor_no_dp"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_sm100_fp8_executor import (
            CutlassSm100ExpertsFp8,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp8PerTensor,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=PureTpRouterFp8PerTensor,
            executor_class=CutlassSm100ExpertsFp8,
            quant_config=quant_config,
        )


class CudaFp8PerTensorSm100EpNormalStrategy(MoeStrategy):
    """SM100 FP8 PerTensor EP normal mode strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(
            config.moe_strategy == "fp8_per_tensor_ep_normal"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_sm100_fp8_executor import (
            CutlassSm100ExpertsFp8,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterFp8PerTensor,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=DeepepNormalRouterFp8PerTensor,
            executor_class=CutlassSm100ExpertsFp8,
            quant_config=quant_config,
        )


class CudaFp8PerTensorSm100EpLowLatencyStrategy(MoeStrategy):
    """SM100 FP8 PerTensor EP low latency strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(
            config.moe_strategy == "fp8_per_tensor_ep_low_latency"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_sm100_fp8_executor import (
            CutlassSm100ExpertsFp8,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=CutlassSm100ExpertsFp8,
            quant_config=quant_config,
        )
