"""Lightweight per-rank dispatch/combine timing recorder using torch.cuda.Event.

Activated by env var MORI_EP_TIMING_DIR=<dir>. When unset, get() returns None
and the router takes the fast path with zero overhead.

Output: one JSONL file per rank at <dir>/mori_ep_timing_wr<rank>.jsonl.
Each line: {"kind": "dispatch"|"combine", "us": float, "idx": int, "rank": int}.
"""
import json
import os
import threading
from collections import deque
from typing import Optional

import torch


class MoriEpTimingRecorder:
    _instance: Optional["MoriEpTimingRecorder"] = None
    _init_lock = threading.Lock()
    _disabled = False

    @classmethod
    def get(cls) -> Optional["MoriEpTimingRecorder"]:
        if cls._disabled:
            return None
        if cls._instance is not None:
            return cls._instance
        with cls._init_lock:
            if cls._disabled:
                return None
            if cls._instance is not None:
                return cls._instance
            out_dir = os.environ.get("MORI_EP_TIMING_DIR", "")
            if not out_dir:
                cls._disabled = True
                return None
            cls._instance = cls(out_dir)
            return cls._instance

    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self._world_rank = int(os.environ.get("WORLD_RANK", "0"))
        self._path = os.path.join(out_dir, f"mori_ep_timing_wr{self._world_rank}.jsonl")
        self._pending = deque()
        self._call_idx = 0
        self._flush_every = int(os.environ.get("MORI_EP_TIMING_FLUSH_EVERY", "10"))
        self._write_lock = threading.Lock()
        # Wipe stale file from previous process; fresh start per benchmark.
        try:
            os.remove(self._path)
        except FileNotFoundError:
            pass

    def record(self, kind: str) -> "_Span":
        return _Span(self, kind)

    def _enqueue(self, start_ev, end_ev, kind: str) -> None:
        self._pending.append((start_ev, end_ev, kind, self._call_idx))
        self._call_idx += 1
        if len(self._pending) >= self._flush_every:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        # Wait for all queued events to complete so elapsed_time is valid.
        torch.cuda.synchronize()
        rows = []
        while self._pending:
            s, e, kind, idx = self._pending.popleft()
            us = s.elapsed_time(e) * 1000.0  # ms -> us
            rows.append({"kind": kind, "us": us, "idx": idx, "rank": self._world_rank})
        with self._write_lock, open(self._path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


class _Span:
    __slots__ = ("rec", "kind", "start_ev", "end_ev")

    def __init__(self, rec: MoriEpTimingRecorder, kind: str):
        self.rec = rec
        self.kind = kind
        self.start_ev = None
        self.end_ev = None

    def __enter__(self):
        self.start_ev = torch.cuda.Event(enable_timing=True)
        self.end_ev = torch.cuda.Event(enable_timing=True)
        self.start_ev.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end_ev.record()
        self.rec._enqueue(self.start_ev, self.end_ev, self.kind)
        return False
