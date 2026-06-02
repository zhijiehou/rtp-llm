import os
from typing import Optional

import torch
import torch.distributed as dist


class DeepEPLatencyTracker:
    """CUDA-event-based latency tracker for DeepEP MoE layer.

    Tracks six phases: forward (total), prepare, dispatch, execute,
    finalize, combine. Aggregates across all MoE layers.

    Env vars:
        DEEPEP_BENCH_LATENCY=1          enable tracking
        DEEPEP_BENCH_LATENCY_INTERVAL   stats print interval (default 50)
        DEEPEP_BENCH_LATENCY_BARRIER=1  dist.barrier() before dispatch/combine
    """

    _instances: list = []
    _global_forward_us: list = []
    _global_prepare_us: list = []
    _global_dispatch_us: list = []
    _global_execute_us: list = []
    _global_finalize_us: list = []
    _global_combine_us: list = []
    _global_token_counts: list = []
    _global_step: int = 0
    _reports_this_step: int = 0

    def __init__(self, router_name: str):
        self._enabled = int(os.environ.get("DEEPEP_BENCH_LATENCY", "0")) == 1
        if not self._enabled:
            return

        self._router_name = router_name
        self._log_interval = int(
            os.environ.get("DEEPEP_BENCH_LATENCY_INTERVAL", "50")
        )
        self._use_barrier = int(
            os.environ.get("DEEPEP_BENCH_LATENCY_BARRIER", "0")
        ) == 1

        self._cur_num_tokens = 0
        self._forward_start: Optional[torch.cuda.Event] = None
        self._prepare_end: Optional[torch.cuda.Event] = None
        self._d_start: Optional[torch.cuda.Event] = None
        self._d_end: Optional[torch.cuda.Event] = None
        self._execute_end: Optional[torch.cuda.Event] = None
        self._c_start: Optional[torch.cuda.Event] = None
        self._c_end: Optional[torch.cuda.Event] = None

        DeepEPLatencyTracker._instances.append(self)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _barrier(self) -> None:
        if self._use_barrier and dist.is_initialized():
            dist.barrier()

    def _record(self) -> torch.cuda.Event:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    # ---- called from FusedMoe.forward() ----

    def mark_forward_start(self) -> None:
        if not self._enabled:
            return
        self._forward_start = self._record()

    def mark_prepare_end(self) -> None:
        if not self._enabled:
            return
        self._prepare_end = self._record()

    def mark_execute_end(self) -> None:
        if not self._enabled:
            return
        self._execute_end = self._record()

    def mark_forward_end(self) -> None:
        if not self._enabled:
            return

        cls = DeepEPLatencyTracker
        num_instances = len(cls._instances)

        if self._cur_num_tokens > 1:
            forward_end = self._record()
            torch.cuda.synchronize()

            cls._global_forward_us.append(
                self._forward_start.elapsed_time(forward_end) * 1000
            )
            cls._global_prepare_us.append(
                self._forward_start.elapsed_time(self._prepare_end) * 1000
            )
            cls._global_dispatch_us.append(
                self._d_start.elapsed_time(self._d_end) * 1000
            )
            cls._global_execute_us.append(
                self._prepare_end.elapsed_time(self._execute_end) * 1000
            )
            cls._global_finalize_us.append(
                self._execute_end.elapsed_time(forward_end) * 1000
            )
            cls._global_combine_us.append(
                self._c_start.elapsed_time(self._c_end) * 1000
            )
            cls._global_token_counts.append(self._cur_num_tokens)

        cls._reports_this_step += 1
        if cls._reports_this_step >= num_instances:
            cls._reports_this_step = 0
            if len(cls._global_forward_us) == 0:
                return
            cls._global_step += 1
            if cls._global_step % self._log_interval == 0:
                self._log_global_stats()

    # ---- called from router ----

    def mark_dispatch_start(self, num_tokens: int = 0) -> None:
        if not self._enabled:
            return
        self._cur_num_tokens = num_tokens
        self._barrier()
        self._d_start = self._record()

    def mark_dispatch_end(self) -> None:
        if not self._enabled:
            return
        self._d_end = self._record()

    def mark_combine_start(self) -> None:
        if not self._enabled:
            return
        self._barrier()
        self._c_start = self._record()

    def mark_combine_end(self) -> None:
        if not self._enabled:
            return
        self._c_end = self._record()

    # ---- stats ----

    @staticmethod
    def _stats(data: list) -> tuple:
        n = len(data)
        if n == 0:
            return 0.0, 0.0, 0.0
        return sum(data) / n, min(data), max(data)

    def _log_global_stats(self) -> None:
        cls = DeepEPLatencyTracker
        n = len(cls._global_forward_us)
        if n == 0:
            return
        num_layers = len(cls._instances)
        tok_avg = sum(cls._global_token_counts) / n

        fwd_avg, fwd_min, fwd_max = self._stats(cls._global_forward_us)
        prep_avg, _, _ = self._stats(cls._global_prepare_us)
        d_avg, d_min, d_max = self._stats(cls._global_dispatch_us)
        exec_avg, _, _ = self._stats(cls._global_execute_us)
        fin_avg, _, _ = self._stats(cls._global_finalize_us)
        c_avg, c_min, c_max = self._stats(cls._global_combine_us)

        rank = dist.get_rank() if dist.is_initialized() else 0
        barrier_tag = " barrier=on" if self._use_barrier else ""
        print(
            f"[{self._router_name} rank={rank} layers={num_layers}]{barrier_tag} "
            f"steps={cls._global_step} tokens_avg={tok_avg:.0f} | "
            f"forward={fwd_avg:.0f}us | "
            f"prepare={prep_avg:.0f}us | "
            f"dispatch: avg={d_avg:.0f}us min={d_min:.0f}us max={d_max:.0f}us | "
            f"execute={exec_avg:.0f}us | "
            f"finalize={fin_avg:.0f}us | "
            f"combine: avg={c_avg:.0f}us min={c_min:.0f}us max={c_max:.0f}us",
            flush=True,
        )
        cls._global_forward_us.clear()
        cls._global_prepare_us.clear()
        cls._global_dispatch_us.clear()
        cls._global_execute_us.clear()
        cls._global_finalize_us.clear()
        cls._global_combine_us.clear()
        cls._global_token_counts.clear()
