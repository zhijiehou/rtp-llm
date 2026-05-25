"""CUDA MOE strategies"""

from .fp8_per_block import (
    CudaFp8PerBlockEpElasticContiguousStrategy,
    CudaFp8PerBlockEpElasticDecodeStrategy,
    CudaFp8PerBlockEpLowLatencyStrategy,
    CudaFp8PerBlockEpNormalStrategy,
    CudaFp8PerBlockNoDPMaskedStrategy,
    CudaFp8PerBlockNoDPStrategy,
)
from .fp8_per_tensor import (
    CudaFp8PerTensorEpElasticContiguousStrategy,
    CudaFp8PerTensorEpElasticDecodeStrategy,
    CudaFp8PerTensorEpLowLatencyStrategy,
    CudaFp8PerTensorEpNormalStrategy,
    CudaFp8PerTensorNoDPStrategy,
)
from .w4a8_int4_per_channel import (
    CudaW4a8Int4PerChannelEpElasticContiguousStrategy,
    CudaW4a8Int4PerChannelEpElasticDecodeStrategy,
    CudaW4a8Int4PerChannelEpLowLatencyStrategy,
    CudaW4a8Int4PerChannelEpNormalStrategy,
    CudaW4a8Int4PerChannelNoDPStrategy,
)
from .no_quant import (
    CudaNoQuantCppStrategy,
    CudaNoQuantDpNormalStrategy,
    CudaNoQuantEpElasticContiguousStrategy,
    CudaNoQuantEpLowLatencyStrategy,
)
from .fp4 import (CudaFp4EpElasticContiguousStrategy,
                  CudaFp4EpLowLatencyStrategy,
                  CudaFp4EpNormalStrategy,
                  CudaFp4NoDPStrategy)


__all__ = [
    # No quantization
    "CudaNoQuantEpLowLatencyStrategy",
    "CudaNoQuantCppStrategy",
    "CudaNoQuantDpNormalStrategy",
    "CudaNoQuantEpElasticContiguousStrategy",
    # FP8 PerBlock
    "CudaFp8PerBlockNoDPMaskedStrategy",
    "CudaFp8PerBlockNoDPStrategy",
    "CudaFp8PerBlockEpLowLatencyStrategy",
    "CudaFp8PerBlockEpNormalStrategy",
    "CudaFp8PerBlockEpElasticContiguousStrategy",
    "CudaFp8PerBlockEpElasticDecodeStrategy",
    # FP8 PerTensor
    "CudaFp8PerTensorNoDPStrategy",
    "CudaFp8PerTensorEpLowLatencyStrategy",
    "CudaFp8PerTensorEpNormalStrategy",
    "CudaFp8PerTensorEpElasticContiguousStrategy",
    "CudaFp8PerTensorEpElasticDecodeStrategy",
    # W4A8 INT4 PerChannel
    "CudaW4a8Int4PerChannelEpLowLatencyStrategy",
    "CudaW4a8Int4PerChannelEpNormalStrategy",
    "CudaW4a8Int4PerChannelNoDPStrategy",
    "CudaW4a8Int4PerChannelEpElasticContiguousStrategy",
    "CudaW4a8Int4PerChannelEpElasticDecodeStrategy",
    # FP4 PerGroup
    "CudaFp4EpLowLatencyStrategy",
    "CudaFp4EpNormalStrategy",
    "CudaFp4NoDPStrategy",
    "CudaFp4EpElasticContiguousStrategy",
]
