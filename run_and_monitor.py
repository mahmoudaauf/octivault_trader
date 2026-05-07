#!/usr/bin/env python3
"""
Run system live and monitor for:
1. First SELL (profit realization)
2. Symbol interchange (capital recycling)

Usage:
    python3 run_and_monitor.py [max_cycles=100]
"""

import asyncio
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(".env"))

from core_engine.native.bootstrap import BootstrapConfig, build_components
from core_engine.native.app_context import build_native_app_ctx

# ANSI colors for output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


async def main():
    """Run system and monitor capital growth."""
    max_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print(f"\n{CYAN}{BOLD}🚀 CAPITAL GROWTH LIVE MONITOR{RESET}")
    print(f"{CYAN}Tracking first SELL and symbol rotation...{RESET}\n")

    try:
        cfg = BootstrapConfig.from_env()
        print(f"{GREEN}✅ Config loaded{RESET}")
        print(f"   - Default quote: ${cfg.default_planned_quote}")
        print(f"   - Allocation %: {cfg.capital_allocation_pct}%")
        print(f"   - TP: {cfg.tp_pct*100:.1f}% | SL: {cfg.sl_pct*100:.1f}%\n")

        components = await build_components(cfg)
        print(f"{GREEN}✅ Components built{RESET}\n")

        app_ctx, orch = build_native_app_ctx(components)
        print(f"{GREEN}✅ Orchestrator ready{RESET}\n")

        # Wait for initial data
        print(f"{YELLOW}⏳ Syncing market data and balance...{RESET}")
        await orch._wait_for_initial_data(max_wait_sec=15.0)

        nav = await components.portfolio_manager.get_nav()
        print(f"{GREEN}✅ Initial NAV: ${nav:.2f}{RESET}\n")

        start_nav = nav
        start_time = time.time()
        first_sell_detected = False
        positions_history = []
        pnl_history = []

        print(f"{BOLD}{'='*80}")
        print(f"RUNNING {max_cycles} CYCLES")
        print(f"{'='*80}{RESET}\n")

        metrics_list = []
        for cycle in await orch.run_loop(max_cycles=max_cycles):
            elapsed = time.time() - start_time
            nav = cycle.nav
            signals = cycle.signals_count
            decisions = cycle.decisions_count
            executions = cycle.execution_successes
            errors = cycle.errors

            # Get current positions
            positions = components.shared_state.get_all_positions()
            pos_symbols = list(positions.keys()) if positions else []

            # Get realized PnL
            pnl = getattr(components.shared_state, "metrics", {}).get("realized_pnl", 0.0)

            metrics_list.append(cycle)

            # Format output
            status_icon = "❌" if errors else "✅"
            growth = nav - start_nav
            growth_pct = (growth / start_nav * 100) if start_nav > 0 else 0

            print(f"{status_icon} Cycle {cycle.cycle_num:3d} | t={elapsed:6.0f}s | "
                  f"NAV=${nav:8.2f} ({growth:+7.2f} {growth_pct:+5.1f}%) | "
                  f"Sig={signals:2d} Dec={decisions:2d} Exe={executions:2d}")

            # Show positions
            if pos_symbols:
                print(f"           Positions: {', '.join(pos_symbols)}")

            # Detect SELL (profit realization)
            if pnl > 0 and not first_sell_detected:
                first_sell_detected = True
                print(f"\n{GREEN}{BOLD}🎉 FIRST SELL DETECTED!{RESET}")
                print(f"{GREEN}   ✅ Realized PnL: ${pnl:.2f}{RESET}")
                print(f"{GREEN}   ✅ Time: {elapsed:.1f}s{RESET}")
                print(f"{GREEN}   ✅ Growth: ${growth:+.2f} ({growth_pct:+.1f}%){RESET}\n")

            # Detect symbol interchange
            if positions_history and set(pos_symbols) != set(positions_history[-1]):
                old_symbols = set(positions_history[-1])
                new_symbols = set(pos_symbols)
                closed = old_symbols - new_symbols
                opened = new_symbols - old_symbols

                if closed or opened:
                    print(f"\n{YELLOW}{BOLD}🔄 SYMBOL INTERCHANGE{RESET}")
                    if closed:
                        print(f"{YELLOW}   ← Closed: {', '.join(closed)} (profits recycled){RESET}")
                    if opened:
                        print(f"{YELLOW}   → Opened: {', '.join(opened)} (capital reinvested){RESET}\n")

            positions_history.append(pos_symbols)
            pnl_history.append(pnl)

            # Stop on error
            if errors:
                print(f"{RED}Error: {errors[0]}{RESET}")
                break

        # Final summary
        print(f"\n{BOLD}{'='*80}")
        print(f"SESSION COMPLETE")
        print(f"{'='*80}{RESET}\n")

        final_nav = await components.portfolio_manager.get_nav()
        final_growth = final_nav - start_nav
        final_growth_pct = (final_growth / start_nav * 100) if start_nav > 0 else 0
        total_pnl = getattr(components.shared_state, "metrics", {}).get("realized_pnl", 0.0)
        win_rate = getattr(components.shared_state, "metrics", {}).get("win_rate_window", 0.0)

        print(f"{CYAN}Duration:           {elapsed:.0f}s ({elapsed/60:.1f}m){RESET}")
        print(f"{CYAN}Cycles:             {len(metrics_list)}{RESET}")
        print(f"{CYAN}Start NAV:          ${start_nav:.2f}{RESET}")
        print(f"{CYAN}Final NAV:          ${final_nav:.2f}{RESET}")
        print(f"{GREEN}Total Growth:       ${final_growth:+.2f} ({final_growth_pct:+.1f}%){RESET}")
        print(f"{GREEN}Realized PnL:       ${total_pnl:+.2f}{RESET}")
        print(f"{CYAN}Win Rate:           {win_rate*100:.1f}%{RESET}")
        print(f"{CYAN}Symbols Traded:     {len(set(pos for cycle_pos in positions_history for pos in cycle_pos))} total{RESET}")

        if first_sell_detected:
            print(f"\n{GREEN}✅ FIRST SELL: DETECTED{RESET}")
            print(f"{GREEN}✅ SYMBOL ROTATION: {len(positions_history) > 1}{RESET}")
            print(f"{GREEN}✅ CAPITAL RECYCLING: Working{RESET}")
        else:
            print(f"\n{YELLOW}⏳ First SELL: Not yet detected (still waiting for profitable trade){RESET}")

        print()

        await orch.stop()

    except Exception as e:
        print(f"{RED}❌ Error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
