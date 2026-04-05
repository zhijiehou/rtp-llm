"""SM100 (Blackwell B200/GB200) optimized FP8 per-tensor executor.

Extends CutlassExpertsFp8 with SM100-specific gating and higher priority.
The actual SM100 tile config dispatch is handled by cutlass_moe_mm_fp8_scaled()
via B200 JSON configs and the SM100 fallback heuristic in fp8_kernel.py.
"""

from typing import Any

from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_moe import (
    CutlassExpertsFp8,
)


class CutlassSm100ExpertsFp8(CutlassExpertsFp8):
    """SM100-optimized FP8 per-tensor executor.

    Same implementation as CutlassExpertsFp8, but with:
    - SM100 hardware gating in check_conditions
    - Higher ExecutorType priority (CUTLASS_SM100_FP8 = 9 > CUTLASS_FP8 = 3)

    This ensures SM100 hardware uses SM100-tuned tile configs (B200 JSONs)
    and the 3-config heuristic (decode/default/2SM) from fp8_kernel.py.
    """

    @classmethod
    def executor_type(cls):
        return ExecutorType.CUTLASS_SM100_FP8

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        from rtp_llm.models_py.utils.arch import get_sm

        super().check_conditions(checker, config)
        checker.check(get_sm()[0] >= 10)
