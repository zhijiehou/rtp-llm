"""CUDA MOE strategies"""

from .fp8_per_block import (
    CudaFp8PerBlockEpLowLatencyStrategy,
    CudaFp8PerBlockEpNormalStrategy,
    CudaFp8PerBlockNoDPMaskedStrategy,
    CudaFp8PerBlockNoDPStrategy,
)
from .fp8_per_tensor import (
    CudaFp8PerTensorEpLowLatencyStrategy,
    CudaFp8PerTensorEpNormalStrategy,
    CudaFp8PerTensorNoDPStrategy,
)
from .w4a8_int4_per_channel import (
    CudaW4a8Int4PerChannelEpLowLatencyStrategy,
    CudaW4a8Int4PerChannelEpNormalStrategy,
    CudaW4a8Int4PerChannelNoDPStrategy,
)
from .no_quant import (
    CudaNoQuantCppStrategy,
    CudaNoQuantDpNormalStrategy,
    CudaNoQuantEpLowLatencyStrategy,
)
from .fp4 import (CudaFp4EpLowLatencyStrategy,
                  CudaFp4EpNormalStrategy,
                  CudaFp4NoDPStrategy)
from .fp8_per_tensor_sm100 import (
    CudaFp8PerTensorSm100EpLowLatencyStrategy,
    CudaFp8PerTensorSm100EpNormalStrategy,
    CudaFp8PerTensorSm100NoDPStrategy,
)
from .fp8_per_block_sm100 import (
    CudaFp8PerBlockSm100EpLowLatencyStrategy,
    CudaFp8PerBlockSm100EpNormalStrategy,
)


__all__ = [
    # No quantization
    "CudaNoQuantEpLowLatencyStrategy",
    "CudaNoQuantCppStrategy",
    "CudaNoQuantDpNormalStrategy",
    # FP8 PerBlock
    "CudaFp8PerBlockNoDPMaskedStrategy",
    "CudaFp8PerBlockNoDPStrategy",
    "CudaFp8PerBlockEpLowLatencyStrategy",
    "CudaFp8PerBlockEpNormalStrategy",
    # FP8 PerTensor
    "CudaFp8PerTensorNoDPStrategy",
    "CudaFp8PerTensorEpLowLatencyStrategy",
    "CudaFp8PerTensorEpNormalStrategy",
    # W4A8 INT4 PerChannel
    "CudaW4a8Int4PerChannelEpLowLatencyStrategy",
    "CudaW4a8Int4PerChannelEpNormalStrategy",
    "CudaW4a8Int4PerChannelNoDPStrategy",
    "CudaFp4EpLowLatencyStrategy",
    "CudaFp4EpNormalStrategy",
    "CudaFp4NoDPStrategy",
    # FP8 PerTensor SM100
    "CudaFp8PerTensorSm100NoDPStrategy",
    "CudaFp8PerTensorSm100EpNormalStrategy",
    "CudaFp8PerTensorSm100EpLowLatencyStrategy",
    # FP8 PerBlock SM100 (FlashInfer groupwise)
    "CudaFp8PerBlockSm100EpNormalStrategy",
    "CudaFp8PerBlockSm100EpLowLatencyStrategy",
]
