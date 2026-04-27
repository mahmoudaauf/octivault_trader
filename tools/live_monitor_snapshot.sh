#!/usr/bin/env bash
set -u

RUN_LOG="${1:-/tmp/octi_run_live_4h_allflags_injector_on_fix_20260421.log}"
OUT_LOG="${2:-/tmp/octi_live_monitor_20260421.log}"
INTERVAL_SEC="${3:-60}"

if [[ ! -f "$RUN_LOG" ]]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR run log not found: $RUN_LOG" >> "$OUT_LOG"
  exit 1
fi

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"

  latest_pnl_line="$(rg '"total_equity"' "$RUN_LOG" | tail -n 1 || true)"
  latest_equity="$(echo "$latest_pnl_line" | sed -n 's/.*"total_equity": \([0-9.]*\).*/\1/p')"
  latest_realized="$(echo "$latest_pnl_line" | sed -n 's/.*"realized_pnl": \([0-9.]*\).*/\1/p')"

  fills_total="$(rg -c 'ORDER_FILLED' "$RUN_LOG" || echo 0)"
  filled_buy="$(rg -c "ORDER_FILLED.*'side': 'BUY'" "$RUN_LOG" || echo 0)"
  filled_sell="$(rg -c 'EM:CLOSE_RESULT.*ok=True.*status=filled' "$RUN_LOG" || echo 0)"

  reject_total="$(rg -c '"event": "TRADE_REJECTED"' "$RUN_LOG" || echo 0)"
  conf_rejects="$(rg -c 'reason=CONF_BELOW_REQUIRED' "$RUN_LOG" || echo 0)"
  sell_edge_blocks="$(rg -c 'EM:SellDynamicGate\] Blocked SELL' "$RUN_LOG" || echo 0)"
  slot_full="$(rg -c 'active_slots=1/1|PORTFOLIO_FULL' "$RUN_LOG" || echo 0)"

  compounding_last="$(rg 'CompoundingEngine' "$RUN_LOG" | tail -n 1 | sed 's/^.*CompoundingEngine - //' || true)"
  loop_last="$(rg 'LOOP_SUMMARY' "$RUN_LOG" | tail -n 1 | sed 's/^.*LOOP_SUMMARY] //' || true)"
  kpi_last="$(rg 'KPI Status' "$RUN_LOG" | tail -n 1 | sed 's/^.*KPI Status: //' || true)"

  echo "[$ts] equity=${latest_equity:-NA} realized_pnl=${latest_realized:-NA} fills_total=$fills_total buy_fills=$filled_buy sell_fills=$filled_sell trade_rejects=$reject_total conf_rejects=$conf_rejects sell_edge_blocks=$sell_edge_blocks slot_full_hits=$slot_full" >> "$OUT_LOG"
  echo "[$ts] loop=$loop_last" >> "$OUT_LOG"
  echo "[$ts] compounding=$compounding_last" >> "$OUT_LOG"
  echo "[$ts] kpi=$kpi_last" >> "$OUT_LOG"
  echo "" >> "$OUT_LOG"

  sleep "$INTERVAL_SEC"
done
