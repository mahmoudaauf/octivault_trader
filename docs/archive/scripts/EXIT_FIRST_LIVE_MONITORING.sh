#!/usr/bin/env bash

# 🎯 EXIT-FIRST STRATEGY: LIVE MONITORING DASHBOARD
# ================================================================================
# Real-time monitoring commands for Exit-First Strategy system

echo "
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║               🚀 EXIT-FIRST LIVE MONITORING DASHBOARD                          ║
║                                                                                ║
║          System: Running Live | Uptime: ~2 min | PID: 61425                    ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
"

echo "
📊 MONITORING OPTIONS
════════════════════════════════════════════════════════════════════════════════
"

echo "
1️⃣  LIVE DASHBOARD (Real-time updates every 5 seconds)
   $ tail -f /tmp/monitor.log

2️⃣  ENTRY GATE VALIDATION (See exit plans being validated)
   $ tail -f logs/system_restart_*.log | grep \"ExitPlan:\"

3️⃣  EXIT MONITORING ACTIVITY (See exits being executed)
   $ tail -f logs/system_restart_*.log | grep \"ExitMonitor:\"

4️⃣  FULL SYSTEM LOG (All system activity)
   $ tail -f logs/system_restart_*.log

5️⃣  EXIT METRICS (See exit distribution tracking)
   $ tail -f logs/system_restart_*.log | grep \"ExitMetrics\"

6️⃣  PROCESS STATUS (Check if system is running)
   $ ps aux | grep -E \"orchestrator|monitor\" | grep python

7️⃣  COUNT ENTRIES (See how many positions entered)
   $ grep -c \"Atomic:BUY\" logs/system_restart_*.log

8️⃣  COUNT EXITS (See how many positions exited)
   $ grep -c \"ExitMonitor:\" logs/system_restart_*.log

9️⃣  EXIT BREAKDOWN (See TP/SL/TIME distribution)
   $ echo \"TP: \$(grep -c 'ExitMonitor:TP' logs/system_restart_*.log) | SL: \$(grep -c 'ExitMonitor:SL' logs/system_restart_*.log) | TIME: \$(grep -c 'ExitMonitor:TIME' logs/system_restart_*.log)\"

🔟 HEALTH CHECK (Test if exit plan validation works)
   $ grep \"ExitPlan:Validate\" logs/system_restart_*.log | head -1
"

echo "
════════════════════════════════════════════════════════════════════════════════

🎯 RECOMMENDED MONITORING SETUP (Open in 2 terminal windows)
════════════════════════════════════════════════════════════════════════════════

TERMINAL 1 (Live Dashboard):
  $ tail -f /tmp/monitor.log

TERMINAL 2 (Exit Activity):
  $ tail -f logs/system_restart_*.log | grep -E \"ExitPlan:|ExitMonitor:\"

════════════════════════════════════════════════════════════════════════════════
"

echo "
✅ SUCCESS CHECKPOINTS (Verify at 5-min intervals)
════════════════════════════════════════════════════════════════════════════════

After 5 minutes:
  ✓ Entry count >= 1:    grep -c \"Atomic:BUY\" logs/system_restart_*.log
  ✓ Validation messages: grep \"ExitPlan:Validate\" logs/system_restart_*.log | wc -l

After 10 minutes:
  ✓ Entry count >= 2:    grep -c \"Atomic:BUY\" logs/system_restart_*.log
  ✓ Exit count >= 1:     grep -c \"ExitMonitor:\" logs/system_restart_*.log

After 30 minutes:
  ✓ Entry count >= 3:    grep -c \"Atomic:BUY\" logs/system_restart_*.log
  ✓ Exit count >= 2:     grep -c \"ExitMonitor:\" logs/system_restart_*.log
  ✓ Distribution:        grep -c 'ExitMonitor:TP' logs/system_restart_*.log
                         grep -c 'ExitMonitor:SL' logs/system_restart_*.log
                         grep -c 'ExitMonitor:TIME' logs/system_restart_*.log

════════════════════════════════════════════════════════════════════════════════
"

echo "
🚨 TROUBLESHOOTING
════════════════════════════════════════════════════════════════════════════════

No validation messages after 5 minutes?
  → Check: tail -100 logs/system_restart_*.log | grep \"ERROR\\|Exception\"
  → Restart: killall python3 && cd /path && nohup python3 orchestrator.py...

No exit monitoring after 10 minutes?
  → Check: grep \"ExitMonitor\" logs/system_restart_*.log | head -5
  → Verify: ps aux | grep orchestrator (should show PID running)

DUST exits appearing?
  → Alert: grep \"DUST\" logs/system_restart_*.log
  → Investigate: These indicate capital getting stuck
  → Action: Check position data with: tail /tmp/monitor.log

Capital deadlock increasing?
  → Check: Balance line in /tmp/monitor.log
  → Compare: To initial balance
  → Action: May need to adjust time exit threshold

════════════════════════════════════════════════════════════════════════════════

System is working when you see:
  ✅ [ExitPlan:Validate] messages (entry gate active)
  ✅ [ExitMonitor:] messages (exit monitoring active)
  ✅ Mix of TP/SL/TIME exits (all pathways working)
  ✅ Capital changing in dashboard (recycling active)

Start with: tail -f /tmp/monitor.log

════════════════════════════════════════════════════════════════════════════════
"
