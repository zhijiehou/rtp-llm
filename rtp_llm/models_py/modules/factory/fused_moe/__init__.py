"""FusedMoe factory module

Uses strategy pattern and builder pattern for refactored MOE factory.

Main components:
- FusedMoeFactory: Main factory class
- MoeStrategy: Strategy base class
- RouterBuilder/ExecutorBuilder: Builder classes
- StrategyRegistry: Strategy registry

Note: DeepEpInitializer is located in rtp_llm.models_py.distributed.deepep_initializer

Usage example:
    from rtp_llm.models_py.modules.factory import FusedMoeFactory

    moe = FusedMoeFactory.create_fused_moe(config, weights)
"""

import torch

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.utils.arch import get_sm, is_cuda

from .defs.fused_moe import FusedMoe
from .factory import FusedMoeFactory
from .strategy_registry import StrategyRegistry

__all__ = ["FusedMoeFactory", "StrategyRegistry", "FusedMoe"]

# ============================================================================
# Device-specific MoE strategy registration
# ============================================================================

device_type = get_device_type()

# Import common strategies
from rtp_llm.models_py.modules.factory.fused_moe.impl.common.strategy.batched_triton_strategy import (
    BatchedTritonStrategy,
)

if device_type == DeviceType.ROCm:
    # ========== ROCm Registry ==========

    # MoE strategies
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.strategy import (
        RocmBf16PureTPStrategy,
        RocmEpLowLatencyStrategy,
        RocmEpNormalStrategy,
        RocmFp8PerBlockPureTPStrategy,
        RocmFp8PerChannelPureTPStrategy,
    )

    registry = StrategyRegistry()
    registry.register(RocmEpLowLatencyStrategy())
    registry.register(RocmEpNormalStrategy())
    registry.register(RocmFp8PerChannelPureTPStrategy())
    registry.register(RocmFp8PerBlockPureTPStrategy())
    registry.register(RocmBf16PureTPStrategy())
    registry.register(BatchedTritonStrategy())
    FusedMoeFactory.set_registry(registry)

else:
    # ========== CUDA Registry ==========

    # MoE strategies
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy import (
        CudaFp8PerBlockEpElasticContiguousStrategy,
        CudaFp8PerBlockEpElasticDecodeStrategy,
        CudaFp8PerBlockEpLowLatencyStrategy,
        CudaFp8PerBlockEpNormalStrategy,
        CudaFp8PerBlockNoDPMaskedStrategy,
        CudaFp8PerBlockNoDPStrategy,
        CudaFp8PerTensorEpElasticContiguousStrategy,
        CudaFp8PerTensorEpElasticDecodeStrategy,
        CudaFp8PerTensorEpLowLatencyStrategy,
        CudaFp8PerTensorEpNormalStrategy,
        CudaFp8PerTensorNoDPStrategy,
        CudaNoQuantCppStrategy,
        CudaNoQuantDpNormalStrategy,
        CudaNoQuantEpElasticContiguousStrategy,
        CudaNoQuantEpLowLatencyStrategy,
        CudaW4a8Int4PerChannelEpElasticContiguousStrategy,
        CudaW4a8Int4PerChannelEpElasticDecodeStrategy,
        CudaW4a8Int4PerChannelEpLowLatencyStrategy,
        CudaW4a8Int4PerChannelEpNormalStrategy,
        CudaW4a8Int4PerChannelNoDPStrategy,
    )

    registry = StrategyRegistry()
    registry.register(CudaFp8PerTensorEpLowLatencyStrategy())
    registry.register(CudaFp8PerTensorEpNormalStrategy())
    registry.register(CudaFp8PerBlockEpLowLatencyStrategy())
    registry.register(CudaFp8PerBlockEpNormalStrategy())
    registry.register(CudaFp8PerBlockNoDPMaskedStrategy())
    registry.register(CudaFp8PerBlockNoDPStrategy())
    registry.register(CudaFp8PerTensorNoDPStrategy())
    registry.register(CudaNoQuantEpLowLatencyStrategy())
    registry.register(CudaNoQuantDpNormalStrategy())
    registry.register(CudaNoQuantCppStrategy())
    registry.register(BatchedTritonStrategy())
    registry.register(CudaW4a8Int4PerChannelEpLowLatencyStrategy())
    registry.register(CudaW4a8Int4PerChannelEpNormalStrategy())
    registry.register(CudaW4a8Int4PerChannelNoDPStrategy())
    # DeepEPv2 elastic variants — gated by USE_DEEPEP_ELASTIC=1 inside
    # DeepEpElasticRouter.check_conditions(), so they never win priority on
    # the default codepath. Two layouts are now registered (each
    # strategy's check_conditions() further gates by the env-driven
    # (do_expand, do_cpu_sync) pair so the two sets are mutually
    # exclusive at resolve time):
    #
    #   *EpElasticContiguousStrategy   — (do_expand=True,  do_cpu_sync=True)
    #     → 2D Contiguous prefill path,routes to the contiguous executor
    #       family (DeepGemmHybrid / CutlassFp8 / CutlassW4a8 / TritonFused
    #       / TrtllmFp4).
    #
    #   *EpElasticDecodeStrategy       — (do_expand=False, do_cpu_sync=False)
    #     → vLLM-style decode cudagraph path (mirrors PR #41183).  Routes
    #       to the subset of contiguous executors that natively guard
    #       ``topk_id == -1`` inside their GPU kernels: CutlassExpertsFp8
    #       (fp8_per_tensor), CutlassExpertsW4a8Int4PerChannel
    #       (w4a8_int4_per_channel),  DeepGemmMaskedExecutorV2
    #       (fp8_per_block — picks V2 over Hybrid to skip the
    #       execute_contiguous L400-401 D2H sanity asserts).
    #
    # Decode-mode coverage is intentionally narrower than contiguous:
    # no_quant (TritonFused clamp silently maps -1 → expert 0) and
    # fp4 (flashinfer trtllm fp4 -1 handling unverified) stay off until
    # the underlying kernels are audited.
    registry.register(CudaNoQuantEpElasticContiguousStrategy())
    registry.register(CudaFp8PerBlockEpElasticContiguousStrategy())
    registry.register(CudaFp8PerBlockEpElasticDecodeStrategy())
    registry.register(CudaFp8PerTensorEpElasticContiguousStrategy())
    registry.register(CudaFp8PerTensorEpElasticDecodeStrategy())
    registry.register(CudaW4a8Int4PerChannelEpElasticContiguousStrategy())
    registry.register(CudaW4a8Int4PerChannelEpElasticDecodeStrategy())
    # Only register FP4 strategies on SM_100+ (and only if CUDA GPU is available)
    if torch.cuda.is_available() and is_cuda() and get_sm()[0] >= 10:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy import (
            CudaFp4EpElasticContiguousStrategy,
            CudaFp4EpLowLatencyStrategy,
            CudaFp4EpNormalStrategy,
            CudaFp4NoDPStrategy,
        )

        registry.register(CudaFp4EpLowLatencyStrategy())
        registry.register(CudaFp4EpNormalStrategy())
        registry.register(CudaFp4NoDPStrategy())
        registry.register(CudaFp4EpElasticContiguousStrategy())
    FusedMoeFactory.set_registry(registry)
