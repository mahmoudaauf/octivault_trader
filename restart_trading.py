#!/usr/bin/env python3
"""
Restart script for live trading system with balance sync fix.
This script:
1. Kills old Python instances
2. Waits for graceful shutdown
3. Starts the trading system fresh
"""

import os
import sys
import time
import subprocess
import signal

def kill_old_instances():
    """Kill any running trading instances."""
    print("=" * 50)
    print("Stopping old instances...")
    print("=" * 50)

    commands = [
        "pkill -f 'python.*main.py' || true",
        "pkill -f 'octivault' || true",
    ]

    for cmd in commands:
        try:
            os.system(cmd)
        except Exception as e:
            print(f"Warning: {e}")

    print("Waiting for graceful shutdown...")
    time.sleep(3)

    # Force kill if needed
    os.system("pkill -9 -f 'python.*main' || true")
    time.sleep(1)
    print("Old instances stopped.\n")

def start_trading_system():
    """Start the live trading system."""
    print("=" * 50)
    print("Starting live trading system...")
    print("=" * 50)

    os_dir = "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
    os.chdir(os_dir)

    print(f"Working directory: {os.getcwd()}")
    print("Running: python3 main.py\n")
    print("=" * 50)
    print("LIVE TRADING SYSTEM OUTPUT:")
    print("=" * 50 + "\n")

    # Start the main trading system
    subprocess.run([sys.executable, "main.py"])

if __name__ == "__main__":
    try:
        kill_old_instances()
        start_trading_system()
    except KeyboardInterrupt:
        print("\n\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)
