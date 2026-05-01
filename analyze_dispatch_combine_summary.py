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

    def fmt(s):
        if s is None: return "  (no samples)"
        return (f"n={s['n']:>3} min={s['min']:6.1f} p50={s['p50']:6.1f} "
                f"mean={s['mean']:6.2f} p99={s['p99']:6.1f} max={s['max']:6.1f}")

    print(f"=== {label} ===\n")

    print("Per-rank breakdown:")
    print(f"  {'rank':>4} | {'dispatch':>55} | {'combine(>12us)':>55}")
    for r in sorted(per_rank):
        d_s = stats(per_rank[r]["dispatch"])
        c_s = stats(per_rank[r]["combine"])
        print(f"   {r:>2}   | {fmt(d_s):>55} | {fmt(c_s):>55}")

    print("\nAggregate:")
    all_d = sum((per_rank[r]["dispatch"] for r in per_rank), [])
    all_c = sum((per_rank[r]["combine"] for r in per_rank), [])
    all_a = sum((per_rank[r]["allreduce"] for r in per_rank), [])
    print(f"     dispatch_kernel: {fmt(stats(all_d))}")
    print(f"  combine_kernel(real): {fmt(stats(all_c))}")
    if all_a:
        print(f"   all_reduce_kernel: {fmt(stats(all_a))}")


if __name__ == "__main__":
    main()
