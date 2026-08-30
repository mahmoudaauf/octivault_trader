#!/bin/bash
# Alert sink for hybrid_supervisor.sh. Receives one argument: the message.
#
# Wired up by being present + executable — hybrid_supervisor.sh defaults
# ALERT_CMD to this file, so no .env plumbing is needed (the supervisor is bash
# and never sources .env, which is why HYBRID_ALERT_CMD sat unset and every
# crash alert went nowhere; the daemon crash-looped 4x on 2026-08-25 unnoticed).
#
# Does two things, both cheap and local:
#   1. appends to logs/hybrid_alerts.log — the durable record
#   2. fires a macOS notification — the thing you actually see
cd "$(dirname "$0")" || exit 0
mkdir -p logs
MSG="${1:-hybrid alert}"
printf '%s [ALERT] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$MSG" >> logs/hybrid_alerts.log

# Never let a notification failure affect the supervisor.
if command -v osascript >/dev/null 2>&1; then
  # Escape double quotes and backslashes for AppleScript's string literal.
  SAFE=$(printf '%s' "$MSG" | sed 's/\\/\\\\/g; s/"/\\"/g')
  osascript -e "display notification \"${SAFE}\" with title \"Hybrid Allocator\" sound name \"Basso\"" \
    >/dev/null 2>&1 || true
fi
exit 0
