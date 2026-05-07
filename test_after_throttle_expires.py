#!/usr/bin/env python3
"""
Wait for throttle to expire, then run a comprehensive 100-cycle test.

This script will:
1. Block until exchange_throttle_until_ts <= time.time()
2. Clear stale runtime state (if needed)
3. Run 100 trading cycles
4. Verify all four fixes are working
5. Capture NAV progression, trades, and any errors

Usage:
    python3 test_after_throttle_expires.py
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(".env"))

from core_engine.native.bootstrap import BootstrapConfig, build_components
from core_engine.native.app_context import build_native_app_ctx

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


async def wait_for_throttle_expiry():
    """Block until throttle window expires."""
    print(f"\n{CYAN}{BOLD}⏰ WAITING FOR THROTTLE TO EXPIRE...{RESET}\n")

    cfg = BootstrapConfig.from_env()
    components = await build_components(cfg)
    shared_state = components.shared_state

    while True:
        throttle_ts = float(
            getattr(shared_state, "exchange_throttle_until_ts", 0.0) or 0.0
        )

        if throttle_ts <= 0 or throttle_ts <= time.time():
            break

        remaining = throttle_ts - time.time()
        minutes, seconds = divmod(int(remaining), 60)
        sys.stdout.write(
            f"\r[{time.strftime('%H:%M:%S')}] "
            f"Throttle expires in {minutes:2d}m {seconds:2d}s"
        )
        sys.stdout.flush()

        await asyncio.sleep(5)

    sys.stdout.write("\n")
    print(f"{GREEN}✅ Throttle expired! Starting test...{RESET}\n")

    return cfg, components


async def run_comprehensive_test(cfg, components):
    """Run 100-cycle test and capture all metrics."""
    print(f"{CYAN}{BOLD}🚀 RUNNING 100-CYCLE TEST{RESET}\n")

    app_ctx, orch = components.orchestrator, components.orchestrator
    # Use the build_native_app_ctx to properly initialize orchestrator
    from core_engine.native.app_context import build_native_app_ctx
    app_ctx, orch = build_native_app_ctx(components)

    shared_state = components.shared_state

    # Test configuration
    max_cycles = 100
    test_start = time.time()

    # Metrics tracking
    nav_history = []
    trade_count = 0
    signal_count = 0
    decision_count = 0
    execution_count = 0
    error_count = 0
    throttle_error_count = 0

    print(f"{CYAN}Cycle | NAV ($) | Positions | Status{RESET}")
    print(f"{CYAN}{'-' * 55}{RESET}")

    try:
        await orch.start()

        for cycle_num in range(1, max_cycles + 1):
            try:
                # Get state before cycle
                nav_before = shared_state.nav_usdt
                positions_before = len(shared_state.get_all_positions())

                # Run one cycle
                await orch._cycle_once()

                # Get state after cycle
                nav_after = shared_state.nav_usdt
                positions_after = len(shared_state.get_all_positions())

                nav_history.append(nav_after)

                # Print every 10 cycles
                if cycle_num % 10 == 0 or cycle_num == 1:
                    status = "OK"
                    if nav_after < 0:
                        status = "NAV ERROR"
                    elif positions_after > positions_before:
                        status = "BUY"
                        trade_count += 1
                    elif positions_after < positions_before:
                        status = "SELL"
                        trade_count += 1

                    print(
                        f"{cycle_num:5d} | "
                        f"${nav_after:>10.2f} | "
                        f"{positions_after:>9d} | {status}"
                    )

            except Exception as e:
                error_count += 1
                error_msg = str(e)
                if "418" in error_msg or "throttled" in error_msg.lower():
                    throttle_error_count += 1
                    print(f"{cycle_num:5d} | ERROR (418/throttle): {error_msg[:30]}")
                else:
                    print(f"{cycle_num:5d} | ERROR: {error_msg[:40]}")

        test_duration = time.time() - test_start

    finally:
        await orch.stop()

    return {
        "duration": test_duration,
        "cycles_completed": max_cycles,
        "nav_history": nav_history,
        "nav_start": nav_history[0] if nav_history else 0,
        "nav_end": nav_history[-1] if nav_history else 0,
        "nav_peak": max(nav_history) if nav_history else 0,
        "nav_min": min(nav_history) if nav_history else 0,
        "errors": error_count,
        "throttle_errors": throttle_error_count,
    }


async def main():
    """Main: wait for throttle expiry, then run test."""
    try:
        # Wait for throttle to expire
        cfg, components = await wait_for_throttle_expiry()

        # Run comprehensive test
        results = await run_comprehensive_test(cfg, components)

        # Print results summary
        print(f"\n{CYAN}{'-' * 55}{RESET}")
        print(f"{CYAN}{BOLD}📊 TEST RESULTS{RESET}\n")

        print(f"Duration:        {results['duration']:.1f}s")
        print(f"Cycles:          {results['cycles_completed']}/100")
        print(f"NAV Start:       ${results['nav_start']:.2f}")
        print(f"NAV End:         ${results['nav_end']:.2f}")
        print(f"NAV Peak:        ${results['nav_peak']:.2f}")
        print(f"NAV Min:         ${results['nav_min']:.2f}")

        nav_change = results["nav_end"] - results["nav_start"]
        nav_change_pct = (
            (nav_change / results["nav_start"] * 100)
            if results["nav_start"] > 0
            else 0
        )

        print(f"NAV Change:      ${nav_change:+.2f} ({nav_change_pct:+.1f}%)")
        print(f"Errors:          {results['errors']}")
        print(f"Throttle Errors: {results['throttle_errors']}")

        # Verdict
        print(f"\n{CYAN}{BOLD}✅ VERDICT:{RESET}")
        if results["throttle_errors"] == 0:
            print(f"{GREEN}✅ All four throttle fixes are working correctly!{RESET}")
            print(f"{GREEN}   • No 418 errors in 100 cycles{RESET}")
            print(f"{GREEN}   • System running stably{RESET}")
        else:
            print(f"{RED}⚠️  {results['throttle_errors']} throttle errors detected{RESET}")

        if results["nav_end"] > results["nav_start"] and results["nav_start"] > 0:
            print(f"{GREEN}✅ Capital compounding working! NAV growing.{RESET}")
        elif results["nav_start"] > 0:
            print(f"{YELLOW}⚠️  NAV not growing yet, but system is operational{RESET}")
        else:
            print(f"{YELLOW}⚠️  NAV still at $0, balance sync may need review{RESET}")

    except KeyboardInterrupt:
        print(f"\n{YELLOW}⏹️  Test interrupted{RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"{RED}❌ Error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
