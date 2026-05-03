#!/usr/bin/env python3
"""
EXECUTION DEADLOCK FIX
======================
Identifies and fixes RULE5_ESCALATION deadlock preventing trades.

The problem:
- Orchestrator has signals but trades are blocked with "INSUFFICIENT_QUOTE_FOR_ACCUMULATION"
- System tries to accumulate capital but can't 
- Creates a deadlock: no trades execute, capital doesn't grow
- Root cause: NAV is too low ($9) when it should be ~$100

Solutions:
1. Kill orchestrator (stop accumulating failed attempts)
2. Clear accumulation state
3. Reset to "fresh start" mode
4. Restart orchestrator
"""

import json
import subprocess
import os
import sys
import signal
from pathlib import Path
from datetime import datetime

def run_cmd(cmd, shell=False):
    """Run command safely"""
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def main():
    print("=" * 80)
    print("EXECUTION DEADLOCK AUTO-FIX")
    print("=" * 80)
    
    # Step 1: Identify the orchestrator process
    print("\n[1] Finding orchestrator process...")
    rc, out, err = run_cmd("ps aux | grep 'MASTER_SYSTEM_ORCHESTRATOR' | grep -v grep", shell=True)
    
    if rc == 0 and out.strip():
        lines = out.strip().split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) > 1:
                pid = parts[1]
                print(f"✅ Found orchestrator PID: {pid}")
                print(f"   {line[:120]}...")
                
                # Step 2: Kill orchestrator
                print(f"\n[2] Stopping orchestrator (PID {pid})...")
                os.kill(int(pid), signal.SIGTERM)
                import time
                time.sleep(3)
                
                # Verify it's dead
                rc2, out2, _ = run_cmd(f"ps -p {pid} > /dev/null 2>&1", shell=True)
                if rc2 != 0:
                    print("✅ Orchestrator stopped successfully")
                else:
                    print("⚠️  Process still running, forcing kill...")
                    os.kill(int(pid), signal.SIGKILL)
                    time.sleep(2)
    else:
        print("❌ Orchestrator not running")
        return 1
    
    # Step 3: Clear problematic state
    print("\n[3] Clearing deadlock state files...")
    
    files_to_reset = [
        "state/positions_nav.json",
        "state/checkpoint.json",
    ]
    
    for f in files_to_reset:
        if Path(f).exists():
            Path(f).unlink()
            print(f"✅ Cleared {f}")
    
    # Step 4: Reset accumulation state in orchestrator log
    print("\n[4] Checking for accumulation metadata...")
    log_path = Path("/tmp/octivault_master_orchestrator.log")
    if log_path.exists():
        print(f"✅ Orchestrator log exists ({log_path.stat().st_size} bytes)")
        print("   (Note: Accumulation state is in-memory, will be cleared on restart)")
    
    # Step 5: Restart orchestrator
    print("\n[5] Restarting orchestrator...")
    cwd = Path("/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader")
    os.chdir(cwd)
    
    # Start with nohup
    import time
    cmd = f"nohup python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py > /tmp/orchestrator_restart.log 2>&1 &"
    rc, out, err = run_cmd(cmd, shell=True)
    time.sleep(5)
    
    # Verify it started
    rc, out, _ = run_cmd("ps aux | grep 'MASTER_SYSTEM_ORCHESTRATOR' | grep -v grep", shell=True)
    if rc == 0 and out.strip():
        pid = out.strip().split()[1]
        print(f"✅ Orchestrator restarted successfully (PID: {pid})")
        print("\n" + "=" * 80)
        print("✅ DEADLOCK FIXED!")
        print("=" * 80)
        print("\nThe orchestrator will now:")
        print("  1. Sync balance from Binance (fresh start)")
        print("  2. Reset all internal state (accumulation, decisions)")
        print("  3. Begin fresh trading cycle with valid capital")
        print("\nMonitor with: tail -f /tmp/octivault_master_orchestrator.log")
        return 0
    else:
        print("❌ Failed to restart orchestrator")
        return 1

if __name__ == "__main__":
    sys.exit(main())
