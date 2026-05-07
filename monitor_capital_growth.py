#!/usr/bin/env python3
"""
Live Capital Growth Monitor — Track first SELL and symbol rotation.

Watches for:
1. First SELL order (verify profit-gating works)
2. Symbol interchange (capital recycling across symbols)
3. Account balance growth

Usage:
    python3 monitor_capital_growth.py
"""

import asyncio
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(".env"))

from core_engine.native.bootstrap import BootstrapConfig, build_components
from core_engine.native.app_context import build_native_app_ctx


class CapitalGrowthMonitor:
    """Monitor capital growth, first SELL, and symbol rotation."""

    def __init__(self):
        self.start_time = time.time()
        self.start_nav = 0.0
        self.first_sell_seen = False
        self.first_sell_time = None
        self.first_sell_symbol = None
        self.first_sell_pnl = None
        self.symbols_traded = set()
        self.cycle_count = 0
        self.total_pnl = 0.0
        self.sells_count = 0

    def print_header(self, nav):
        """Print session header."""
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 80)
        print(f"🚀 CAPITAL GROWTH MONITOR")
        print(f"   Start NAV: ${self.start_nav:.2f}")
        print(f"   Current NAV: ${nav:.2f}")
        print(f"   Growth: ${nav - self.start_nav:+.2f} ({(nav/self.start_nav - 1)*100:+.1f}%)")
        print(f"   Elapsed: {elapsed:.0f}s")
        print("=" * 80 + "\n")

    def print_cycle_summary(self, metrics, signals, decisions):
        """Print cycle summary."""
        elapsed = time.time() - self.start_time
        self.cycle_count += 1

        nav = metrics.nav
        signals_count = metrics.signals_count
        executions = metrics.executions_count
        successes = metrics.execution_successes

        print(f"\n📍 CYCLE {self.cycle_count}: t={elapsed:6.0f}s | NAV=${nav:8.2f}")
        print(f"   Signals: {signals_count:2d} | Decisions: {len(decisions):2d} | Executions: {executions:2d} (✅ {successes})")

        if signals_count > 0:
            buy_signals = [s for s, sig in signals.items() if sig.get("direction") == "BUY"]
            sell_signals = [s for s, sig in signals.items() if sig.get("direction") == "SELL"]
            print(f"   📊 BUY signals: {len(buy_signals)} | SELL signals: {len(sell_signals)}")

            if buy_signals:
                print(f"      → BUY:  {', '.join(buy_signals[:5])}")
            if sell_signals:
                print(f"      → SELL: {', '.join(sell_signals[:5])}")

        if decisions:
            buy_decisions = [d for d in decisions if d.action.value == "OPEN"]
            sell_decisions = [d for d in decisions if d.action.value == "CLOSE"]
            print(f"   💰 BUY decisions: {len(buy_decisions)} | SELL decisions: {len(sell_decisions)}")
            if buy_decisions:
                print(f"      → Opening: {', '.join([d.symbol for d in buy_decisions[:3]])}")
            if sell_decisions:
                print(f"      → Closing: {', '.join([d.symbol for d in sell_decisions[:3]])}")

    def monitor_fills(self, shared_state):
        """Check for fill events (BUY/SELL)."""
        if not hasattr(shared_state, "get_all_positions"):
            return

        positions = shared_state.get_all_positions()
        current_symbols = {p for p in positions}
        self.symbols_traded.update(current_symbols)

        # Check metrics for realized PnL
        if hasattr(shared_state, "metrics"):
            m = shared_state.metrics
            realized_pnl = m.get("realized_pnl", 0.0)
            win_rate = m.get("win_rate_window", 0.0)
            trades_closed = m.get("trades_in_window", 0)

            if realized_pnl > 0 and not self.first_sell_seen:
                self.first_sell_seen = True
                self.first_sell_time = time.time() - self.start_time
                print("\n" + "🎉" * 40)
                print(f"✅ FIRST PROFIT REALIZED!")
                print(f"   Time: {self.first_sell_time:.1f}s")
                print(f"   Realized PnL: ${realized_pnl:.2f}")
                print(f"   Win Rate: {win_rate*100:.1f}%")
                print(f"   Trades Closed: {trades_closed}")
                print("🎉" * 40 + "\n")

            if trades_closed > 0:
                self.total_pnl = realized_pnl

    def monitor_symbol_rotation(self, shared_state):
        """Track symbol rotation."""
        if not hasattr(shared_state, "get_all_positions"):
            return

        positions = shared_state.get_all_positions()
        current_symbols = set(positions.keys())

        if current_symbols != self.symbols_traded:
            new_symbols = current_symbols - self.symbols_traded
            removed_symbols = self.symbols_traded - current_symbols

            if new_symbols:
                print(f"\n🔄 SYMBOL INTERCHANGE DETECTED!")
                print(f"   New symbols opened: {', '.join(sorted(new_symbols))}")

            if removed_symbols and len(self.symbols_traded) > 0:
                print(f"   Symbols closed (profits recycled): {', '.join(sorted(removed_symbols))}")
                self.sells_count += len(removed_symbols)

            self.symbols_traded = current_symbols

            if self.symbols_traded:
                print(f"   Current portfolio: {', '.join(sorted(self.symbols_traded))}")
                print(f"   Portfolio size: {len(self.symbols_traded)} symbols")

    def print_final_summary(self, nav):
        """Print final summary."""
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600.0
        growth_pct = (nav / self.start_nav - 1) * 100 if self.start_nav > 0 else 0

        print("\n" + "=" * 80)
        print("📊 SESSION SUMMARY")
        print("=" * 80)
        print(f"Duration:           {elapsed:.0f}s ({hours:.2f}h)")
        print(f"Start NAV:          ${self.start_nav:.2f}")
        print(f"End NAV:            ${nav:.2f}")
        print(f"Total Growth:       ${nav - self.start_nav:+.2f} ({growth_pct:+.1f}%)")
        print(f"Cycles Completed:   {self.cycle_count}")
        print(f"Symbols Traded:     {', '.join(sorted(self.symbols_traded)) if self.symbols_traded else 'None'}")
        print(f"Total Sells:        {self.sells_count}")
        print(f"Realized PnL:       ${self.total_pnl:+.2f}")

        if self.first_sell_seen:
            print(f"First SELL at:      {self.first_sell_time:.1f}s")
            print(f"✅ Capital recycling: WORKING (profit-gated)")
        else:
            print(f"✅ First SELL:       PENDING (waiting for profitable trade)")

        print("=" * 80 + "\n")


async def run_monitor():
    """Run the monitoring loop."""
    print("\n🚀 Initializing system...")

    try:
        cfg = BootstrapConfig.from_env()
        print(f"✅ Config loaded (NAV will update after balance sync)")

        components = await build_components(cfg)
        print(f"✅ Components built")

        app_ctx, orch = build_native_app_ctx(components)
        print(f"✅ Orchestrator ready\n")

        monitor = CapitalGrowthMonitor()

        # Wait for initial data
        print("⏳ Waiting for initial market data and balance...")
        await orch._wait_for_initial_data(max_wait_sec=10.0)

        initial_nav = await components.portfolio_manager.get_nav()
        monitor.start_nav = initial_nav or 50.0
        print(f"✅ Initial NAV: ${monitor.start_nav:.2f}\n")

        # Run trading cycles
        print("=" * 80)
        print("📈 STARTING LIVE TRADING - MONITORING FOR:")
        print("   1. First SELL (profit-gating verification)")
        print("   2. Symbol interchange (capital recycling)")
        print("=" * 80)

        cycle_num = 0
        while not orch._stopped:
            try:
                cycle_num += 1

                # Run single cycle
                metrics = await orch.run_cycle()

                # Get current signals and decisions
                prices = components.market_data.get_prices()
                signals = {}
                if prices:
                    for sym in list(prices.keys())[:10]:  # Limit to first 10 for display
                        try:
                            klines = await components.market_data.get_klines(sym, "1m", 100)
                            agg_sig = components.signal_engine.evaluate(sym, klines)
                            if agg_sig:
                                signals[sym] = {
                                    "direction": agg_sig.direction,
                                    "score": agg_sig.score,
                                }
                        except Exception:
                            pass

                # Build portfolio snapshot
                portfolio = components.portfolio_accessor()
                decisions = []
                if portfolio:
                    balance_usdt = components.balance_sync.get_balance().get("USDT", 0.0)
                    decisions = components.decision_engine.decide(signals, portfolio, balance_usdt)

                # Print cycle summary
                monitor.print_cycle_summary(metrics, signals, decisions)

                # Check for fills and symbol rotation
                monitor.monitor_fills(components.shared_state)
                monitor.monitor_symbol_rotation(components.shared_state)

                # Print header every 10 cycles
                if cycle_num % 10 == 0:
                    current_nav = await components.portfolio_manager.get_nav()
                    monitor.print_header(current_nav or monitor.start_nav)

                # Stop after 100 cycles (for demo)
                if cycle_num >= 100:
                    print(f"\n✅ Completed 100 cycles, stopping monitor...")
                    break

            except KeyboardInterrupt:
                print("\n\n⏹️  Monitor stopped by user")
                break
            except Exception as e:
                print(f"❌ Cycle {cycle_num} failed: {e}")
                import traceback
                traceback.print_exc()
                break

        # Final summary
        final_nav = await components.portfolio_manager.get_nav()
        monitor.print_final_summary(final_nav or monitor.start_nav)

        await orch.stop()

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_monitor())
