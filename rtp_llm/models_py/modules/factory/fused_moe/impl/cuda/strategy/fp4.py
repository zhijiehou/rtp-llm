"""CUDA FP4 Per-Group quantization strategies"""

import os
from typing import Any, Dict

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class CudaFp4NoDPStrategy(MoeStrategy):
    """CUDA FP4 PerGroup single GPU strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_strategy == "fp4_no_dp" or config.moe_strategy == "auto")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.trtllm_fp4_executor import (
            TrtllmFp4Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp4PerGroup,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8,
            block_shape=[16, 16],
        )
        return StrategyAttributes(
            router_class=PureTpRouterFp4PerGroup,
            executor_class=TrtllmFp4Executor,
            quant_config=quant_config,
        )


class CudaFp4EpLowLatencyStrategy(MoeStrategy):
    """CUDA FP4 PerGroup EP low latency strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "modelopt_fp4")
        checker.check(config.moe_strategy == "fp4_ep_low_latency" or config.moe_strategy == "auto")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp4_executor import (
            CutedslFp4Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8,
            block_shape=[16, 16],
        )
        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=CutedslFp4Executor,
            quant_config=quant_config,
        )


class CudaFp4EpNormalStrategy(MoeStrategy):
    """CUDA FP4 PerGroup EP normal mode strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_strategy == "fp4_ep_normal" or config.moe_strategy == "auto")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.trtllm_fp4_executor import (
            TrtllmFp4Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterFp4PerGroup,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8,
            block_shape=[16, 16],
        )
        return StrategyAttributes(
            router_class=DeepepNormalRouterFp4PerGroup,
            executor_class=TrtllmFp4Executor,
            quant_config=quant_config,
        )


class CudaFp4EpElasticContiguousStrategy(MoeStrategy):
    """CUDA FP4 PerGroup EP elastic 2D Contiguous strategy.

    Selected when ``USE_DEEPEP_ELASTIC=1`` with the default
    ``DEEPEP_ELASTIC_DO_EXPAND=1, DEEPEP_ELASTIC_DO_CPU_SYNC=1`` —
    pairs the elastic router (tight ``[ΣN_e, hidden]`` layout) with
    ``TrtllmFp4Executor``.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "modelopt_fp4")
        do_expand = bool(int(os.environ.get("DEEPEP_ELASTIC_DO_EXPAND", "1")))
        do_cpu_sync = bool(int(os.environ.get("DEEPEP_ELASTIC_DO_CPU_SYNC", "1")))
        checker.check(do_expand and do_cpu_sync)
        checker.check(
            config.moe_strategy == "fp4_ep_elastic_contiguous"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.trtllm_fp4_executor import (
            TrtllmFp4Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_elastic_router import (
            DeepEpElasticRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8,
            block_shape=[16, 16],
        )
        return StrategyAttributes(
            router_class=DeepEpElasticRouter,
            executor_class=TrtllmFp4Executor,
            quant_config=quant_config,
        )
