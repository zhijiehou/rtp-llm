"""CUDA W4A8 INT4 PerChannel quantization strategies"""

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
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class CudaW4a8Int4PerChannelNoDPStrategy(MoeStrategy):
    """CUDA W4A8 INT4 PerChannel single GPU strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method
            in ("W4A8_INT4_PER_CHANNEL", "W4A8_INT4_PER_CHANNEL_COMPRESSED")
        )
        checker.check(config.moe_strategy == "w4a8_int4_per_channel_no_dp" or config.moe_strategy == "auto")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_w4a8_moe import (
            CutlassExpertsW4a8Int4PerChannel,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterW4a8Int4PerChannel,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=PureTpRouterW4a8Int4PerChannel,
            executor_class=CutlassExpertsW4a8Int4PerChannel,
            quant_config=quant_config,
        )


class CudaW4a8Int4PerChannelEpLowLatencyStrategy(MoeStrategy):
    """CUDA W4A8 INT4 PerChannel EP low latency strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method
            in ("W4A8_INT4_PER_CHANNEL", "W4A8_INT4_PER_CHANNEL_COMPRESSED")
        )
        checker.check(config.moe_strategy == "w4a8_int4_per_channel_ep_low_latency" or config.moe_strategy == "auto")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_w4a8_moe import (
            CutlassBatchedExpertsW4a8Int4PerChannel,
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
            executor_class=CutlassBatchedExpertsW4a8Int4PerChannel,
            quant_config=quant_config,
        )


class CudaW4a8Int4PerChannelEpNormalStrategy(MoeStrategy):
    """CUDA W4A8 INT4 PerChannel EP normal mode strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method
            in ("W4A8_INT4_PER_CHANNEL", "W4A8_INT4_PER_CHANNEL_COMPRESSED")
        )
        checker.check(config.moe_strategy == "w4a8_int4_per_channel_ep_normal" or config.moe_strategy == "auto")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_w4a8_moe import (
            CutlassExpertsW4a8Int4PerChannel,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterW4a8Int4PerChannel,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=DeepepNormalRouterW4a8Int4PerChannel,
            executor_class=CutlassExpertsW4a8Int4PerChannel,
            quant_config=quant_config,
        )


class CudaW4a8Int4PerChannelEpElasticContiguousStrategy(MoeStrategy):
    """CUDA W4A8 INT4 PerChannel EP elastic 2D Contiguous strategy.

    Selected when ``USE_DEEPEP_ELASTIC=1`` with the default
    ``DEEPEP_ELASTIC_DO_EXPAND=1, DEEPEP_ELASTIC_DO_CPU_SYNC=1`` —
    pairs the elastic router (tight ``[ΣN_e, hidden]`` layout) with
    ``CutlassExpertsW4a8Int4PerChannel``.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method
            in ("W4A8_INT4_PER_CHANNEL", "W4A8_INT4_PER_CHANNEL_COMPRESSED")
        )
        do_expand = bool(int(os.environ.get("DEEPEP_ELASTIC_DO_EXPAND", "1")))
        do_cpu_sync = bool(int(os.environ.get("DEEPEP_ELASTIC_DO_CPU_SYNC", "1")))
        checker.check(do_expand and do_cpu_sync)
        checker.check(
            config.moe_strategy == "w4a8_int4_per_channel_ep_elastic_contiguous"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_w4a8_moe import (
            CutlassExpertsW4a8Int4PerChannel,
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
            executor_class=CutlassExpertsW4a8Int4PerChannel,
            quant_config=quant_config,
        )


class CudaW4a8Int4PerChannelEpElasticDecodeStrategy(MoeStrategy):
    """CUDA W4A8 INT4 PerChannel EP elastic decode cudagraph strategy.

    Selected when ``USE_DEEPEP_ELASTIC=1`` with
    ``DEEPEP_ELASTIC_DO_EXPAND=0, DEEPEP_ELASTIC_DO_CPU_SYNC=0`` —
    pairs the elastic router decode-mode layout
    (``[worst_case_N, hidden]`` + per-row ``recv_topk_idx`` with ``-1``
    sentinels) with ``CutlassExpertsW4a8Int4PerChannel``.

    Shares the same ``-1``-handling story as the FP8 PerTensor decode
    strategy: ``cutlass_w4a8_moe.py:169`` collapses ``topk_id == -1`` to
    the no-expert slot via ``torch.where(topk_ids != -1, topk_ids,
    self.E)`` and reuses the Triton pre/post reorder kernels.  The
    router places ``expert_tokens_meta=None`` so the executor falls
    through to the no-D2H ``num_gemm_tokens = topk_ids.numel()`` branch.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method
            in ("W4A8_INT4_PER_CHANNEL", "W4A8_INT4_PER_CHANNEL_COMPRESSED")
        )
        do_expand = bool(int(os.environ.get("DEEPEP_ELASTIC_DO_EXPAND", "1")))
        do_cpu_sync = bool(int(os.environ.get("DEEPEP_ELASTIC_DO_CPU_SYNC", "1")))
        checker.check((not do_expand) and (not do_cpu_sync))
        checker.check(
            config.moe_strategy == "w4a8_int4_per_channel_ep_elastic_decode"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_w4a8_moe import (
            CutlassExpertsW4a8Int4PerChannel,
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
            executor_class=CutlassExpertsW4a8Int4PerChannel,
            quant_config=quant_config,
        )
