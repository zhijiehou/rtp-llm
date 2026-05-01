#!/usr/bin/env python3
"""Compare dispatch + combine kernel duration in RTP-LLM vs MoriEP standalone bench.

Usage: python analyze_dispatch_combine_summary.py <out_dir> [label]
"""
import json, re, sys
from pathlib import Path

DISPATCH_PAT = re.compile(r"EpDispatchIntraNodeKernel")
COMBINE_PAT = re.compile(r"EpCombineIntraNodeKernel")
ALLREDUCE_PAT = re.compile(r"all_reduce|AllReduce|ncclAllReduce|rcclAllReduce")


def main():
    # 待解析的路径名
    out_dir = Path(sys.argv[1])
    # 输出标题的标签
    label = sys.argv[2] if len(sys.argv) > 2 else out_dir.name
    # 获取profiler文件
    files = sorted(out_dir.glob("profiler_wr*_*.json"))

    per_rank = {}  # rank -> dict(dispatch=[], combine=[], allreduce=[])
    for f in files:
        m = re.search(r"wr(\d+)_", f.name)
        rank = int(m.group(1)) if m else -1
        with open(f) as fp:
            data = json.load(fp)
        events = data.get("traceEvents", data) if isinstance(data, dict) else data
        d_list, c_list, a_list = [], [], []
        for ev in events:
            if not isinstance(ev, dict): continue
            if "ts" not in ev or "dur" not in ev: continue
            n = ev.get("name", "")
            cat = ev.get("cat", "")
            d = ev["dur"]
            if DISPATCH_PAT.search(n): d_list.append(d)
            elif COMBINE_PAT.search(n) and d > 12: c_list.append(d)  # filter trivial
            elif ALLREDUCE_PAT.search(n) and "kernel" in cat.lower(): a_list.append(d)
        per_rank[rank] = dict(dispatch=d_list, combine=c_list, allreduce=a_list)

    def stats(lst):
        if not lst: return None
        s = sorted(lst); n = len(s)
        return dict(n=n, min=s[0], p50=s[n//2], mean=sum(s)/n,
                    p99=s[int(n*0.99)], max=s[-1])

    def row(s):
        if s is None:
            return f"{'-':>4} {'-':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>6}"
        return (f"{s['n']:>4} {s['min']:>6.1f} {s['p50']:>6.1f} "
                f"{s['mean']:>6.2f} {s['p99']:>6.1f} {s['max']:>6.1f}")

    col_hdr = f"{'n':>4} {'min':>6} {'p50':>6} {'mean':>6} {'p99':>6} {'max':>6}"
    col_w = len(col_hdr)
    sep = "-" * (4 + 3 + col_w + 3 + col_w)

    print(f"=== {label} ===\n")
    print("Per-rank breakdown (us):")
    print(f"{'':>4} | {'dispatch':^{col_w}} | {'combine (>12us)':^{col_w}}")
    print(f"{'rank':>4} | {col_hdr} | {col_hdr}")
    print(sep)
    for r in sorted(per_rank):
        d_s = stats(per_rank[r]["dispatch"])
        c_s = stats(per_rank[r]["combine"])
        print(f"{r:>4} | {row(d_s)} | {row(c_s)}")
    print(sep)

    all_d = sum((per_rank[r]["dispatch"] for r in per_rank), [])
    all_c = sum((per_rank[r]["combine"] for r in per_rank), [])
    all_a = sum((per_rank[r]["allreduce"] for r in per_rank), [])
    print(f"{'ALL':>4} | {row(stats(all_d))} | {row(stats(all_c))}")

    if all_a:
        print(f"\nall_reduce_kernel (us): {row(stats(all_a))}")


if __name__ == "__main__":
    main()
