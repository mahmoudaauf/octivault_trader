#!/usr/bin/env python3
"""
🎯 FLEXIBLE DURATION TRADING SESSION
====================================

Simple wrapper that runs the master orchestrator for any duration with
real-time output monitoring and automated logging.

USAGE:
    export APPROVE_LIVE_TRADING=YES
    python3 RUN_3HOUR_SESSION.py [--paper] [--duration HOURS]
    
    Default: 3 hours
    Example: python3 RUN_3HOUR_SESSION.py --paper --duration 10
    
Exit codes:
    0 = Success
    1 = Failed
"""

import asyncio
import logging
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import time
import signal

# ============================================================================
# MAIN RUNNER
# ============================================================================

async def run_trading_session(duration_hours: int = 3, paper_mode: bool = False):
    """Run trading session for specified duration"""
    
    project_root = Path(__file__).parent
    orchestrator_path = project_root / "🎯_MASTER_SYSTEM_ORCHESTRATOR.py"
    
    # Build command
    cmd = [
        sys.executable,
        str(orchestrator_path),
        "--duration", str(duration_hours)
    ]
    
    if paper_mode:
        cmd.append("--paper")
    
    # Setup environment
    env = os.environ.copy()
    env["APPROVE_LIVE_TRADING"] = "YES"
    env["PYTHONUNBUFFERED"] = "1"
    
    print("\n" + "=" * 100)
    print("🎯 STARTING 3-HOUR TRADING SESSION")
    print("=" * 100)
    print(f"Start Time: {datetime.now()}")
    print(f"Duration: 3 hours")
    print(f"Mode: {'📝 PAPER TRADING' if paper_mode else '🔴 LIVE TRADING'}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 100 + "\n")
    
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=3)
    
    try:
        # Start subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(project_root)
        )
        
        # Monitor output
        print("🟢 System running... (monitoring for 3 hours)\n")
        
        line_count = 0
        last_status_time = time.time()
        
        while process.returncode is None:
            try:
                # Read line with timeout
                line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=1.0
                )
                
                if line:
                    line_count += 1
                    output = line.decode().rstrip()
                    print(output)
                    
                    # Print elapsed time every 60 lines or every minute
                    now = time.time()
                    elapsed = (datetime.now() - start_time).total_seconds()
                    
                    if line_count % 60 == 0 or (now - last_status_time) >= 60:
                        elapsed_hours = elapsed / 3600
                        remaining = (end_time - datetime.now()).total_seconds()
                        remaining_hours = remaining / 3600
                        
                        status = f"\n⏱️  [{int(elapsed_hours)}h {int((elapsed % 3600) / 60)}m elapsed | {int(remaining_hours)}h {int((remaining % 3600) / 60)}m remaining]\n"
                        print(status)
                        last_status_time = now
                
                # Check if we've exceeded 3 hours
                if (datetime.now() - start_time).total_seconds() > 3 * 3600:
                    print("\n✅ 3-hour session complete!")
                    process.terminate()
                    try:
                        await asyncio.wait_for(
                            process.wait(),
                            timeout=10
                        )
                    except asyncio.TimeoutError:
                        process.kill()
                    break
                
            except asyncio.TimeoutError:
                # No output, just check if process is still running
                if process.returncode is not None:
                    break
                continue
            except Exception as e:
                print(f"Error reading output: {e}")
                break
        
        # Wait for process to finish
        try:
            returncode = await asyncio.wait_for(
                process.wait(),
                timeout=30
            )
        except asyncio.TimeoutError:
            process.kill()
            returncode = -1
        
        print("\n" + "=" * 100)
        print("✅ SESSION COMPLETE")
        print("=" * 100)
        print(f"End Time: {datetime.now()}")
        print(f"Total Duration: {(datetime.now() - start_time).total_seconds() / 3600:.2f} hours")
        print(f"Exit Code: {returncode}")
        print("=" * 100 + "\n")
        
        return returncode == 0
        
    except KeyboardInterrupt:
        print("\n🛑 Session interrupted by user")
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except:
            process.kill()
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ============================================================================
# MAIN
# ============================================================================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run trading session for custom duration")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode")
    parser.add_argument("--duration", type=int, default=3, help="Session duration in hours (default: 3)")
    args = parser.parse_args()
    
    # Check permissions for live trading
    if not args.paper and os.getenv("APPROVE_LIVE_TRADING") != "YES":
        print("❌ Live trading requires: export APPROVE_LIVE_TRADING=YES")
        sys.exit(1)
    
    # Run session
    success = await run_trading_session(duration_hours=args.duration, paper_mode=args.paper)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
