"""CUDA FP8 PerTensor quantization strategies"""

import os
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


class CudaFp8PerTensorEpLowLatencyStrategy(MoeStrategy):
    """CUDA FP8 PerTensor EP low latency strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_strategy == "fp8_per_tensor_ep_low_latency" or config.moe_strategy == "auto")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_moe import (
            CutlassBatchedExpertsFp8,
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
            executor_class=CutlassBatchedExpertsFp8,
            quant_config=quant_config,
        )


class CudaFp8PerTensorEpNormalStrategy(MoeStrategy):
    """CUDA FP8 PerTensor EP normal mode strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_strategy == "fp8_per_tensor_ep_normal" or config.moe_strategy == "auto")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_moe import (
            CutlassExpertsFp8,
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
            executor_class=CutlassExpertsFp8,
            quant_config=quant_config,
        )


class CudaFp8PerTensorNoDPStrategy(MoeStrategy):
    """CUDA FP8 PerTensor single GPU strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_strategy == "fp8_per_tensor_no_dp" or config.moe_strategy == "auto")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_moe import (
            CutlassExpertsFp8,
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
            executor_class=CutlassExpertsFp8,
            quant_config=quant_config,
        )


class CudaFp8PerTensorEpElasticContiguousStrategy(MoeStrategy):
    """CUDA FP8 PerTensor EP elastic 2D Contiguous strategy.

    Selected when ``USE_DEEPEP_ELASTIC=1`` with the default
    ``DEEPEP_ELASTIC_DO_EXPAND=1, DEEPEP_ELASTIC_DO_CPU_SYNC=1`` —
    pairs the elastic router with ``CutlassExpertsFp8``.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        do_expand = bool(int(os.environ.get("DEEPEP_ELASTIC_DO_EXPAND", "1")))
        do_cpu_sync = bool(int(os.environ.get("DEEPEP_ELASTIC_DO_CPU_SYNC", "1")))
        checker.check(do_expand and do_cpu_sync)
        checker.check(
            config.moe_strategy == "fp8_per_tensor_ep_elastic_contiguous"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_moe import (
            CutlassExpertsFp8,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_elastic_router import (
            DeepEpElasticRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=DeepEpElasticRouter,
            executor_class=CutlassExpertsFp8,
            quant_config=quant_config,
        )


class CudaFp8PerTensorEpElasticDecodeStrategy(MoeStrategy):
    """CUDA FP8 PerTensor EP elastic decode cudagraph strategy.

    Selected when ``USE_DEEPEP_ELASTIC=1`` with
    ``DEEPEP_ELASTIC_DO_EXPAND=0, DEEPEP_ELASTIC_DO_CPU_SYNC=0`` —
    pairs the elastic router decode-mode layout
    (``[worst_case_N, hidden]`` + per-row ``recv_topk_idx`` with ``-1``
    sentinels) with ``CutlassExpertsFp8``.

    ``CutlassExpertsFp8`` natively maps ``topk_id == -1`` to a
    "no-expert" sentinel via ``local_topk_ids = torch.where(topk_ids
    != -1, topk_ids, E)`` (``cutlass_moe.py:158``) and the underlying
    pre/post reorder Triton kernels guard ``expert_id < num_local_experts``
    per row.  The router places ``expert_tokens_meta=None`` on the
    payload so ``cutlass_moe.py:128-131`` falls through to the
    ``num_gemm_tokens = topk_ids.numel()`` branch without any D2H.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        do_expand = bool(int(os.environ.get("DEEPEP_ELASTIC_DO_EXPAND", "1")))
        do_cpu_sync = bool(int(os.environ.get("DEEPEP_ELASTIC_DO_CPU_SYNC", "1")))
        checker.check((not do_expand) and (not do_cpu_sync))
        checker.check(
            config.moe_strategy == "fp8_per_tensor_ep_elastic_decode"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_moe import (
            CutlassExpertsFp8,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_elastic_router import (
            DeepEpElasticRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=DeepEpElasticRouter,
            executor_class=CutlassExpertsFp8,
            quant_config=quant_config,
        )
