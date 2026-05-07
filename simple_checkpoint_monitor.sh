#!/bin/bash
# Simple checkpoint monitor - tracks NAV milestones from live_run.log

TARGETS=(100 110 125 150 200)
REACHED=()
BASELINE=0
CHECKPOINT_FILE="checkpoints_simple.jsonl"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "CHECKPOINT MONITOR — Tracking NAV Profit Compounding"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

while true; do
    # Get latest NAV from logs
    LATEST_NAV=$(tail -100 live_run.log 2>/dev/null | grep "nav=" | tail -1 | sed 's/.*nav=\s*\([0-9.]*\).*/\1/' | awk '{print $1}')

    if [ -z "$LATEST_NAV" ] || [ "$LATEST_NAV" == "0.00" ] || [ "$LATEST_NAV" == "0" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏳ Waiting for balance data (API throttle)... NAV still $LATEST_NAV"
        sleep 10
        continue
    fi

    # Set baseline on first positive NAV
    if [ "$BASELINE" == "0" ] || [ "$BASELINE" == "0.0" ]; then
        BASELINE=$(echo "$LATEST_NAV" | awk '{printf "%.2f", $1}')
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📊 BASELINE NAV DETECTED: \$$BASELINE"
        echo ""
    fi

    # Check each target
    for TARGET in "${TARGETS[@]}"; do
        # Check if already reached
        if [[ ! " ${REACHED[@]} " =~ " ${TARGET} " ]]; then
            # Check if current NAV >= target
            if (( $(echo "$LATEST_NAV >= $TARGET" | bc -l) )); then
                REACHED+=($TARGET)
                GAIN=$(echo "scale=2; ($LATEST_NAV - $BASELINE) / $BASELINE * 100" | bc -l)
                TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

                echo ""
                echo "[${TIMESTAMP}] 🎯 CHECKPOINT REACHED: \$${TARGET}"
                echo "    Actual NAV: \$$LATEST_NAV"
                echo "    Gain: ${GAIN}%"
                echo "    Time: $(tail -1 live_run.log | awk '{print $1, $2}')"
                echo ""

                # Save to file
                echo "{\"timestamp\":\"${TIMESTAMP}\",\"target\":${TARGET},\"actual\":${LATEST_NAV},\"gain_pct\":${GAIN}}" >> "$CHECKPOINT_FILE"
            fi
        fi
    done

    # Print status every 30 seconds
    CYCLES=$(tail -5 live_run.log 2>/dev/null | grep "cycle" | tail -1 | grep -o "cycle [0-9]*")
    SIGS=$(tail -5 live_run.log 2>/dev/null | grep "sigs=" | tail -1 | grep -o "sigs=\s*[0-9]*" | grep -o "[0-9]*$")
    EXE=$(tail -5 live_run.log 2>/dev/null | grep "exe=" | tail -1 | grep -o "exe=\s*[0-9]*" | grep -o "[0-9]*$")

    if [ $(($(date +%s) % 30)) -lt 2 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Status: NAV=\$$LATEST_NAV | ${CYCLES} | signals=${SIGS:-0} | executions=${EXE:-0}"
    fi

    sleep 5
done
