#!/bin/bash
URL="http://localhost:10666/v1/chat/completions"
WARMUP=2
RUNS=50
CONCURRENCY=4
PROMPT_LENS=(4096)

make_prompt() {
    local target=$1
    local unit="The quick brown fox jumps over the lazy dog and cat. "
    local repeat=$((target / 12))
    printf "%0.s${unit}" $(seq 1 $repeat)
}

echo "=== Concurrent Prefill TTFT Benchmark ==="
echo "URL: $URL"
echo "Concurrency: $CONCURRENCY, Warmup: $WARMUP, Runs: $RUNS"
echo ""
printf "%-15s %-12s %-12s %-12s %s\n" \
    "Prompt_tokens" "TTFT_p50" "TTFT_min" "TTFT_max" "Prefill_speed"
echo "--------------------------------------------------------------"

for pl in "${PROMPT_LENS[@]}"; do
    PROMPT=$(make_prompt $pl)
    PAYLOAD=$(cat <<EOF
{
    "temperature": 0,
    "max_tokens": 1,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [
        {"role": "user", "content": "$PROMPT"}
    ]
}
EOF
)

    TMPFILE=$(mktemp /tmp/bench_payload_XXXXXX.json)
    echo "$PAYLOAD" > "$TMPFILE"

    # Warmup
    for i in $(seq 1 $WARMUP); do
        for c in $(seq 1 $CONCURRENCY); do
            curl -s --max-time 300 "$URL" \
                -H "Content-Type: application/json" \
                -d @"$TMPFILE" > /dev/null 2>&1 &
        done
        wait
    done

    # Runs
    RESULTS_DIR=$(mktemp -d /tmp/bench_results_XXXXXX)

    for i in $(seq 1 $RUNS); do
        for c in $(seq 1 $CONCURRENCY); do
            (
                RESP=$(curl -s --max-time 300 "$URL" \
                    -H "Content-Type: application/json" \
                    -d @"$TMPFILE")
                TTFT=$(echo "$RESP" | python -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d['aux_info']['first_token_cost_time'])
except:
    print(-1)
" 2>/dev/null)
                PT=$(echo "$RESP" | python -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d['usage']['prompt_tokens'])
except:
    print(0)
" 2>/dev/null)
                echo "${TTFT},${PT}" >> "$RESULTS_DIR/run_${i}_${c}.txt"
            ) &
        done
        wait
    done

    rm -f "$TMPFILE"

    # Aggregate
    python -c "
import os, sys
results_dir = '$RESULTS_DIR'
ttfts = []
pt = 0
for f in os.listdir(results_dir):
    with open(os.path.join(results_dir, f)) as fh:
        for line in fh:
            parts = line.strip().split(',')
            t = float(parts[0])
            p = int(parts[1])
            if t > 0:
                ttfts.append(t)
            if p > 0 and pt == 0:
                pt = p
if not ttfts:
    print('pt=~%-5d  FAILED' % $pl)
    sys.exit()
ttfts.sort()
n = len(ttfts)
p50 = ttfts[n // 2]
mn = ttfts[0]
mx = ttfts[-1]
speed = pt * 1000 / p50 if p50 > 0 else 0
print('%-15d %-12.1f %-12.1f %-12.1f %.0f' % (pt, p50, mn, mx, speed))
"

    rm -rf "$RESULTS_DIR"
done

echo ""
echo "=== Done ==="
