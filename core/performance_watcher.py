
import asyncio
import logging
import json
import os
import time
from typing import Optional, Dict, Any

class PerformanceWatcher:
    """
    Monitors trading performance against a target hourly ROI (e.g. $10/hr).
    Adjusts system parameters (Dynamic Tuning) if objectives are not met.
    """
    def __init__(self, shared_state, config, logger: Optional[logging.Logger] = None):
        self.shared_state = shared_state
        self.config = config
        self.logger = logger or logging.getLogger("PerformanceWatcher")
        
        # User Objective: $10/hr from ~$268 capital (massive ~3.7%/hr target)
        self.target_hourly_pnl = float(self._cfg("TARGET_HOURLY_PNL", 10.0))
        self.check_interval = float(self._cfg("PERF_CHECK_INTERVAL", 300)) # Check every 5 mins
        
        self._running = False
        self._task = None
        self._start_equity = 0.0
        self._start_time = 0.0
        
        self.tuning_dir = "tuned_params"
        os.makedirs(self.tuning_dir, exist_ok=True)

    def _cfg(self, key, default):
        return getattr(self.config, key, default)

    async def start(self):
        if self._running: return
        self._running = True
        self._start_time = time.time()
        self._start_equity = 0.0 # Will be populated asynchronously
        self.logger.info(f"PerformanceWatcher started. Target: ${self.target_hourly_pnl}/hr. Waiting for equity sync...")
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _get_total_equity(self) -> float:
        """Fetch total estimated equity (USDT + Asset Value) from SharedState."""
        try:
            if hasattr(self.shared_state, "get_total_dashboard_equity"):
                return await self.shared_state.get_total_dashboard_equity()
            
            # Fallback manual calculation
            balances = await self.shared_state.get_all_balances() # {sym: {free, locked}}
            total = 0.0
            for sym, bal in balances.items():
                qty = float(bal.get('free', 0)) + float(bal.get('locked', 0))
                if qty > 0:
                    if sym == 'USDT':
                        total += qty
                    else:
                        price = await self.shared_state.get_latest_price(f"{sym}USDT")
                        if price:
                            total += qty * price
            return total
        except Exception as e:
            self.logger.error(f"Failed to calc equity: {e}")
            return 0.0

    async def _watch_loop(self):
        while self._running:
            try:
                # 1. Acquire Baseline if missing
                if self._start_equity <= 0:
                    eq = await self._get_total_equity()
                    if eq > 0:
                        self._start_equity = eq
                        self._start_time = time.time() # Reset clock to now
                        self.logger.info(f"PerformanceWatcher Baseline Set: ${self._start_equity:.2f} at {time.ctime(self._start_time)}")
                    else:
                        # Wait and retry sooner if no baseline yet
                        await asyncio.sleep(5.0)
                        continue

                # 2. Regular Evaluation
                await asyncio.sleep(self.check_interval)
                await self._evaluate_performance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in watch loop: {e}", exc_info=True)

    async def _evaluate_performance(self):
        current_equity = await self._get_total_equity()
        elapsed_hours = (time.time() - self._start_time) / 3600.0
        
        if elapsed_hours < 0.1: return # Too soon
        
        pnl = current_equity - self._start_equity
        actual_hourly_rate = pnl / elapsed_hours
        
        gap = self.target_hourly_pnl - actual_hourly_rate
        
        self.logger.info(f"Performance Report: PnL=${pnl:.2f} | Rate=${actual_hourly_rate:.2f}/hr | Target=${self.target_hourly_pnl}/hr")
        
        # Log to SharedState for UI/Dashboard
        if hasattr(self.shared_state, "emit_event"):
            await self.shared_state.emit_event("PerformanceUpdate", {
                "pnl": pnl,
                "hourly_rate": actual_hourly_rate,
                "target_gap": gap,
                "equity": current_equity
            })

        # --- Dynamic Tuning Logic ---
        # If we are failing significantly (rate < 50% of target) and not losing money
        if pnl < self.target_hourly_pnl:
             self.logger.info("Underperforming target. Attempting dynamic parameter adjustments (Aggressive Mode)...")
             report = {
                "pnl": pnl,
                "target": self.target_hourly_pnl,
                "hourly_rate": actual_hourly_rate,
                "drawdown": (self._start_equity - current_equity) / self._start_equity if self._start_equity > 0 else 0.0
             }
             await self._tune_agents(report)

        # If we are losing money rapidly (drawdown > 2%)
        elif self._start_equity > 0 and current_equity < (self._start_equity * 0.98):
             self.logger.warning("Significant drawdown detected. Switching to Defensive Mode.")
             report = {
                "pnl": pnl,
                "target": self.target_hourly_pnl,
                "hourly_rate": actual_hourly_rate,
                "drawdown": (self._start_equity - current_equity) / self._start_equity
             }
             await self._tune_agents(report)

    async def _tune_agents(self, report: Dict[str, Any]):
        """
        Adjust agent knobs based on performance report.
        Push live updates to SharedState.dynamic_config.
        """
        pnl = float(report.get("pnl", 0.0))
        target = float(report.get("target", 0.0))
        drawdown = float(report.get("drawdown", 0.0))

        aggressiveness = 0.5  # default
        if pnl < target:
            # Underperforming: increase aggressiveness if PnL is positive but below target
            if pnl > 0:
                aggressiveness = 0.7
            else:
                aggressiveness = 0.9  # Aggressive recovery
        
        if drawdown > 0.02:
            aggressiveness = 0.3  # Defensive mode for > 2% drawdown

        self.logger.info(f"Performance Report: PnL=${pnl:.2f} | Rate=${report.get('hourly_rate', 0.0):.2f}/hr | Target=${target:.1f}/hr")
        
        # 1. Base Trading Knobs
        min_conf = 0.65
        max_trade = 100.0

        if aggressiveness >= 0.7:
            self.logger.info("🚀 Performance below target. Switching to Aggressive Recovery.")
            min_conf = 0.50  # Lower threshold to take more trades
            max_trade = 150.0 # Increase per-trade limit
        elif aggressiveness < 0.5:
            self.logger.warning("🚨 Significant drawdown detected. Switching to Defensive Mode.")
            min_conf = 0.75  # Tighten signal requirement
            max_trade = 50.0  # Reduce exposure
        
        overrides = {
            "MIN_SIGNAL_CONF": min_conf,
            "MAX_PER_TRADE_USDT": max_trade
        }

        # 2. Capital Management Knobs (Phase 5)
        # Rebalancing frequency: faster in aggressive mode, slower in defensive
        rebalance_interval = 300
        compound_threshold = 10.0

        if aggressiveness >= 0.7:
            rebalance_interval = 120  # Rebalance more frequently to capture trends
            compound_threshold = 5.0   # Compound smaller gains
        elif aggressiveness < 0.5:
            rebalance_interval = 900  # Slower rebalancing to save fees/slippage
            compound_threshold = 20.0  # Be more picky with compounding

        overrides["REBALANCE_INTERVAL_SEC"] = rebalance_interval
        overrides["COMPOUNDING_THRESHOLD"] = compound_threshold

        # 3. Risk Management Knobs
        # Tighten risk limits during drawdowns
        if aggressiveness < 0.5:
            overrides["MAX_DRAWDOWN_PCT"] = 0.05  # Tighten to 5%
            overrides["MAX_DAILY_LOSS_PCT"] = 0.02 # Tighten to 2%
        else:
            # Revert to defaults (or whatever is in static config)
            overrides["MAX_DRAWDOWN_PCT"] = 0.10 
            overrides["MAX_DAILY_LOSS_PCT"] = 0.05

        # Push to SharedState
        if hasattr(self.shared_state, "update_dynamic_config"):
            try:
                await self.shared_state.update_dynamic_config(overrides)
                self.logger.info(f"Dynamic Tuning Applied: {', '.join([f'{k}={v}' for k, v in overrides.items()])}")
            except Exception as e:
                self.logger.error(f"Failed to push dynamic config: {e}")

        # Persistence
        try:
            with open(self.tuning_file, "w") as f:
                json.dump(overrides, f, indent=2)
        except Exception as e:
            self.logger.debug(f"Could not persist tuning: {e}")
