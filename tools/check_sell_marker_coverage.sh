#!/usr/bin/env bash
set -euo pipefail

# Verify that each exchange SELL order ID has at least one bot-side marker
# in app log and/or trade journal logs.
#
# Usage:
#   tools/check_sell_marker_coverage.sh [missing_sells.csv]
#
# Input CSV format:
#   SYMBOL,ORDER_ID

EX_IDS_FILE="${1:-missing_sells.csv}"
APP_LOG="${APP_LOG:-logs/app.log}"
MARKER_PAT="${MARKER_PAT:-SELL_ORDER_PLACED|RECONCILED_DELAYED_FILL|ORDER_FILLED|EM:CLOSE_RESULT|EM:DelayedFill|TPSL:CLOSE_RESULT|TRUTH_AUDIT_FILL_RECOVERED|TruthAuditor:SellFinalize|liquidation_exit_result}"

if [[ ! -f "$EX_IDS_FILE" ]]; then
  # Try a common fallback name pattern.
  alt_file="$(ls -1 missing_sells*.csv 2>/dev/null | head -n 1 || true)"
  if [[ -n "${alt_file:-}" && -f "$alt_file" ]]; then
    EX_IDS_FILE="$alt_file"
  else
    echo "Missing exchange SELL id file: $EX_IDS_FILE"
    echo "Expected format: SYMBOL,ORDER_ID"
    exit 1
  fi
fi

declare -a BOT_FILES=()
[[ -f "$APP_LOG" ]] && BOT_FILES+=("$APP_LOG")
for f in logs/trade_journal_*.jsonl; do
  [[ -e "$f" ]] && BOT_FILES+=("$f")
done

if [[ ${#BOT_FILES[@]} -eq 0 ]]; then
  echo "No bot logs found (expected $APP_LOG and/or logs/trade_journal_*.jsonl)"
  exit 1
fi

# Normalize exchange SELL order IDs:
# - second CSV column
# - keep numeric IDs only
cut -d',' -f2 "$EX_IDS_FILE" \
  | tr -d '\r' \
  | rg -a '^[0-9]+$' \
  | sort -u > /tmp/exchange_sell_ids.txt

: > /tmp/sell_ids_matched.txt
: > /tmp/sell_ids_unmatched.txt

while IFS= read -r oid; do
  if rg -aF "$oid" "${BOT_FILES[@]}" 2>/dev/null | rg -a -e "$MARKER_PAT" >/dev/null 2>&1; then
    echo "$oid" >> /tmp/sell_ids_matched.txt
  else
    echo "$oid" >> /tmp/sell_ids_unmatched.txt
  fi
done < /tmp/exchange_sell_ids.txt

sort -u -o /tmp/sell_ids_matched.txt /tmp/sell_ids_matched.txt
sort -u -o /tmp/sell_ids_unmatched.txt /tmp/sell_ids_unmatched.txt

echo "INPUT_FILE=$EX_IDS_FILE"
echo "APP_LOG_PRESENT=$([[ -f "$APP_LOG" ]] && echo 1 || echo 0)"
echo "EXCHANGE_SELL_IDS_TOTAL=$(wc -l < /tmp/exchange_sell_ids.txt | tr -d ' ')"
echo "MATCHED_IDS=$(wc -l < /tmp/sell_ids_matched.txt | tr -d ' ')"
echo "UNMATCHED_IDS=$(wc -l < /tmp/sell_ids_unmatched.txt | tr -d ' ')"

if [[ -s /tmp/sell_ids_unmatched.txt ]]; then
  echo "UNMATCHED_ORDER_IDS:"
  cat /tmp/sell_ids_unmatched.txt
fi

echo
echo "MARKER SAMPLE (tail):"
rg -an -e "$MARKER_PAT" "${BOT_FILES[@]}" | tail -n 80 || true

