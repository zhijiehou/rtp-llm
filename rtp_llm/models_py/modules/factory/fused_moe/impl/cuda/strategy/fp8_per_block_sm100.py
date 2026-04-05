"""CUDA FP8 PerBlock SM100 (Blackwell) strategies.

These strategies gate on SM100 hardware and use FlashInferFp8GroupwiseExecutor
which provides 2SM MMA for large prefill batches via FlashInfer's
group_gemm_fp8_nt_groupwise kernel.

Only EP strategies are provided here — the FlashInfer executor uses
ep_scatter/ep_gather for flat token reordering, which is the format
produced by DeepEP routers. For NoDPStrategy (PureTpRouter with 3D
batched layout), continue using the DeepGEMM-based executors.

On non-SM100 hardware or when FlashInfer is unavailable, these strategies
will not match, and the existing DeepGEMM-based per-block strategies are used.
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


class CudaFp8PerBlockSm100EpNormalStrategy(MoeStrategy):
    """SM100 FP8 PerBlock EP normal mode strategy using FlashInfer groupwise."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(
            config.moe_strategy == "fp8_per_block_ep_normal"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.flashinfer_fp8_groupwise_executor import (
            FlashInferFp8GroupwiseExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterFp8PerBlock,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=DeepepNormalRouterFp8PerBlock,
            executor_class=FlashInferFp8GroupwiseExecutor,
            quant_config=quant_config,
        )


class CudaFp8PerBlockSm100EpLowLatencyStrategy(MoeStrategy):
    """SM100 FP8 PerBlock EP low latency strategy using FlashInfer groupwise."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(
            config.moe_strategy == "fp8_per_block_ep_low_latency"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.flashinfer_fp8_groupwise_executor import (
            FlashInferFp8GroupwiseExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=FlashInferFp8GroupwiseExecutor,
            quant_config=quant_config,
        )
