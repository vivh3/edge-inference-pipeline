#!/usr/bin/env bash
# Unattended sustained-load run: start it, walk away, read the artifacts later.
#
# The measurement is twelve minutes of steady state and the interesting failure
# modes are slow ones, so sitting in front of four terminals is a poor use of
# the time. This starts the graph, waits for the first published record before
# the clock starts (the model takes ~100 s to load, and the budget row is about
# steady state, not startup), stops everything cleanly, and commits the result.
#
#   tools/run_sustained.sh [--minutes 12] [--push]
#
# tegrastats is NOT started here: it needs sudo, and an unattended script that
# waits on a password prompt is worse than one that does less. Start it
# yourself before running this, and stop it afterwards:
#
#   sudo tegrastats --interval 1000 --logfile /tmp/tegrastats-sustained.log
#   sudo tegrastats --stop
#
# Leaving it running past the end is harmless: the summariser splits busy from
# idle, so trailing idle samples only make the brackets clearer.

set -u -o pipefail

MINUTES=12
PUSH=0
while [ $# -gt 0 ]; do
    case "$1" in
        --minutes) MINUTES="$2"; shift 2 ;;
        --push) PUSH=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/results/sustained"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

# One place for the per-shell setup, so this and a human terminal agree.
# shellcheck disable=SC1090,SC1091
source "$REPO/env.sh" > /dev/null
cd "$REPO/ros2_ws"

PIDS=()

stop_nodes() {
    # Reverse launch order: inference first so it stops submitting, telemetry
    # last so it is still subscribed when the final record is published and
    # can write its summary on the way out.
    for i in $(seq $((${#PIDS[@]} - 1)) -1 0); do
        kill -INT "${PIDS[$i]}" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
# Ctrl-C or a kill should still shut the nodes down, so an aborted run does not
# leave three processes holding the camera and 5 GB of GPU.
trap 'echo; echo "interrupted, stopping nodes"; stop_nodes; exit 130' INT TERM

echo "== launching, logs in $LOGS"
python3 -m edge_perception.telemetry_node --ros-args -p out_dir:="$OUT" \
    > "$LOGS/telemetry_node.log" 2>&1 &
PIDS+=($!)
python3 -m edge_perception.capture_node > "$LOGS/capture_node.log" 2>&1 &
PIDS+=($!)
python3 -m edge_perception.inference_node --ros-args -p use_mock_engine:=false \
    > "$LOGS/inference_node.log" 2>&1 &
PIDS+=($!)

# Wait for the first record rather than a fixed sleep: weight loading took 78 s
# one run and 101 s another, and timing from launch would put startup inside
# the measurement by a different amount each time.
echo "== waiting for the first published record (model load takes ~100 s)"
CSV="$OUT/results.csv"
for _ in $(seq 1 600); do
    # Header plus at least one row.
    if [ -f "$CSV" ] && [ "$(wc -l < "$CSV")" -gt 1 ]; then break; fi
    # A node that died takes the run with it; no point waiting out the timeout.
    for pid in "${PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "a node exited before publishing; see $LOGS" >&2
            stop_nodes
            exit 1
        fi
    done
    sleep 1
done
if [ ! -f "$CSV" ] || [ "$(wc -l < "$CSV")" -le 1 ]; then
    echo "no record published within 600 s; see $LOGS" >&2
    stop_nodes
    exit 1
fi

echo "== first record in, measuring for $MINUTES minutes"
sleep $((MINUTES * 60))

echo "== stopping"
stop_nodes
sleep 2

git -C "$REPO" add -A results/sustained
if git -C "$REPO" diff --cached --quiet; then
    echo "== nothing to commit"
else
    git -C "$REPO" commit -q -m "Sustained ${MINUTES}-minute run on a re-pointed camera

Unattended, via tools/run_sustained.sh. Artifacts only; the write-up follows
once the numbers have been read."
    if [ "$PUSH" = "1" ]; then
        for delay in 0 2 4 8 16; do
            [ "$delay" = "0" ] || sleep "$delay"
            if git -C "$REPO" push -u origin claude/edge-multimodal-robotics-wbb4z9; then
                echo "== pushed"
                break
            fi
            echo "== push failed, retrying"
        done
    fi
fi

echo
echo "== summary"
python3 - "$OUT/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
age = s.get("result_age") or {}
first, last = s["first_minute_result_age_p50"], s["last_minute_result_age_p50"]
print(f"records            {s['records']} ({s['unparseable']} unparseable)")
print(f"duration           {s['duration_s']:.0f} s")
print(f"result age p50     {age.get('p50', float('nan')):.3f} s  (budget 6.5)")
print(f"result age p90     {age.get('p90', float('nan')):.3f} s  (budget 6.6)")
if first and last:
    print(f"first vs last min  {first:.3f} -> {last:.3f} s  "
          f"({100 * abs(last - first) / first:.2f}%, budget 10%)")
rate = s.get("invalid_output_rate")
if rate is not None:
    print(f"invalid output     {100 * rate:.1f}%  (budget 35%)")
print(f"failures           {s['failures'] or 'none'}")
PY
echo
echo "Remember to: sudo tegrastats --stop"
