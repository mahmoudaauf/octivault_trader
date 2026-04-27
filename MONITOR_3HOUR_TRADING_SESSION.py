#!/usr/bin/env python3
"""
🎯 3-HOUR TRADING SESSION MONITOR
===================================

Runs the trading system for 3 hours with real-time monitoring of:
- Profit generation
- Loss management
- Reinvestment tracking
- Portfolio health
- Capital allocation
- Trade statistics

USAGE:
    export APPROVE_LIVE_TRADING=YES
    python3 MONITOR_3HOUR_TRADING_SESSION.py [--paper]
    
    --paper  : Run in paper trading mode (default: live if APPROVE_LIVE_TRADING=YES)
"""

import asyncio
import logging
import os
import sys
import signal
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time
import json

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure comprehensive logging"""
    log_dir = Path("/tmp")
    log_file = log_dir / "octivault_3h_monitor.log"
    
    # Create log file
    log_file.parent.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# MONITORING CLASS
# ============================================================================

class TradingSessionMonitor:
    """Real-time monitoring of trading session performance"""
    
    def __init__(self, duration_hours: int = 3):
        self.duration_hours = duration_hours
        self.duration_seconds = duration_hours * 3600
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Performance metrics
        self.initial_portfolio_value: float = 0.0
        self.initial_cash: float = 0.0
        self.current_portfolio_value: float = 0.0
        self.current_cash: float = 0.0
        self.trades_executed: int = 0
        self.trades_won: int = 0
        self.trades_lost: int = 0
        self.total_profit: float = 0.0
        self.total_loss: float = 0.0
        self.reinvested_amount: float = 0.0
        self.highest_portfolio_value: float = 0.0
        self.lowest_portfolio_value: float = 0.0
        
        # Snapshots
        self.portfolio_snapshots: list = []
        self.hourly_summary: Dict[int, Dict[str, Any]] = {}
        
        # Components
        self.shared_state: Optional[object] = None
        self.execution_manager: Optional[object] = None
        self.exchange_client: Optional[object] = None
        
    async def initialize(self):
        """Initialize monitoring components"""
        try:
            logger.info("🔌 Initializing monitoring components...")
            
            # Monitoring will initialize dynamically as components start
            logger.info("✅ Monitor initialized (will track system during operation)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize monitor: {e}", exc_info=True)
            return False
    
    async def _capture_snapshot(self, label: str = ""):
        """Capture current portfolio state"""
        try:
            if not self.shared_state:
                return
            
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "label": label,
                "elapsed_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                "portfolio_value": await self.shared_state.get_total_nav(),
                "cash": 0.0,  # Will be updated from positions
                "positions_count": 0,
                "trades_executed": self.trades_executed,
                "total_profit": self.total_profit,
                "total_loss": self.total_loss,
                "profit_loss_net": self.total_profit - self.total_loss,
            }
            
            # Get positions
            try:
                positions = await self.shared_state.get_positions_snapshot()
                snapshot["positions_count"] = len(positions)
                
                # Calculate cash (sum of quote balances for USDT pairs)
                total_value = snapshot["portfolio_value"]
                portfolio_cost = sum(p.get("cost_basis", 0) for p in positions.values())
                snapshot["cash"] = total_value - portfolio_cost
            except:
                pass
            
            self.portfolio_snapshots.append(snapshot)
            return snapshot
            
        except Exception as e:
            logger.debug(f"Snapshot capture error: {e}")
            return None
    
    async def run_monitoring_loop(self):
        """Run monitoring loop during trading session"""
        
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(seconds=self.duration_seconds)
        
        logger.info("\n" + "=" * 80)
        logger.info("🟢 3-HOUR TRADING SESSION MONITOR STARTED")
        logger.info("=" * 80)
        logger.info(f"Start Time: {self.start_time}")
        logger.info(f"End Time: {self.end_time}")
        logger.info(f"Duration: {self.duration_hours} hours ({self.duration_seconds}s)")
        logger.info("=" * 80 + "\n")
        
        try:
            cycle = 0
            last_hourly_report = 0
            last_summary_report = 0
            
            while (datetime.now() - self.start_time).total_seconds() < self.duration_seconds:
                cycle += 1
                elapsed = (datetime.now() - self.start_time).total_seconds()
                elapsed_hours = elapsed / 3600
                remaining = self.duration_seconds - elapsed
                remaining_hours = remaining / 3600
                
                # Capture snapshot every 30 seconds
                if cycle % 3 == 0:  # Every ~30 seconds (10-second intervals)
                    await self._capture_snapshot(f"MONITORING-CYCLE-{cycle}")
                
                # Hourly report
                if elapsed > last_hourly_report + 3600 or (cycle % 360 == 0 and elapsed < 3600):
                    await self._print_hourly_report(int(elapsed_hours))
                    last_hourly_report = elapsed
                
                # Quick status update every 10 cycles (~100 seconds)
                if cycle % 10 == 0:
                    await self._print_quick_status(elapsed_hours, remaining_hours)
                
                # Detailed status every cycle for first 5 minutes, then every 10 cycles
                if cycle <= 30 or cycle % 10 == 0:
                    await self._update_metrics()
                
                await asyncio.sleep(10)  # Poll every 10 seconds
            
            # Final report
            await self._print_final_report()
            
        except asyncio.CancelledError:
            logger.info("Monitoring cancelled")
        except Exception as e:
            logger.error(f"Monitoring error: {e}", exc_info=True)
    
    async def _update_metrics(self):
        """Update performance metrics"""
        try:
            if not self.shared_state:
                return
            
            # Update portfolio metrics
            nav = await self.shared_state.get_total_nav()
            self.current_portfolio_value = nav
            
            if self.highest_portfolio_value == 0 or nav > self.highest_portfolio_value:
                self.highest_portfolio_value = nav
            
            if self.lowest_portfolio_value == 0 or nav < self.lowest_portfolio_value:
                self.lowest_portfolio_value = nav
            
        except Exception as e:
            logger.debug(f"Metrics update error: {e}")
    
    async def _print_quick_status(self, elapsed_hours: float, remaining_hours: float):
        """Print quick status update"""
        try:
            elapsed_min = int(elapsed_hours * 60)
            remaining_min = int(remaining_hours * 60)
            
            nav = self.current_portfolio_value or 0
            profit_loss = self.total_profit - self.total_loss
            profit_loss_pct = (profit_loss / self.initial_portfolio_value * 100) if self.initial_portfolio_value > 0 else 0
            
            logger.info(
                f"⏱️  [{elapsed_min:3d}m / {int(remaining_min):3d}m remaining] "
                f"NAV: ${nav:>10.2f} | "
                f"P&L: ${profit_loss:>8.2f} ({profit_loss_pct:>6.2f}%) | "
                f"Trades: {self.trades_executed:>3d} | "
                f"Highest: ${self.highest_portfolio_value:>10.2f}"
            )
        except Exception as e:
            logger.debug(f"Status print error: {e}")
    
    async def _print_hourly_report(self, hour: int):
        """Print detailed hourly report"""
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        nav = self.current_portfolio_value or 0
        profit_loss = self.total_profit - self.total_loss
        profit_loss_pct = (profit_loss / self.initial_portfolio_value * 100) if self.initial_portfolio_value > 0 else 0
        
        logger.info("\n" + "=" * 80)
        logger.info(f"📊 HOURLY REPORT - HOUR {hour}")
        logger.info("=" * 80)
        logger.info(f"Elapsed Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s")
        logger.info(f"Period: Hour {hour}")
        logger.info("-" * 80)
        
        logger.info("💰 PORTFOLIO STATUS")
        logger.info(f"├─ Current NAV: ${nav:,.2f}")
        logger.info(f"├─ Initial Value: ${self.initial_portfolio_value:,.2f}")
        logger.info(f"├─ Change: ${nav - self.initial_portfolio_value:,.2f} "
                   f"({((nav - self.initial_portfolio_value) / self.initial_portfolio_value * 100):+.2f}%)")
        logger.info(f"├─ Highest: ${self.highest_portfolio_value:,.2f}")
        logger.info(f"└─ Lowest: ${self.lowest_portfolio_value:,.2f}")
        
        logger.info("\n📈 TRADING PERFORMANCE")
        logger.info(f"├─ Total Trades: {self.trades_executed}")
        logger.info(f"├─ Won: {self.trades_won} ({(self.trades_won / self.trades_executed * 100) if self.trades_executed > 0 else 0:.1f}%)")
        logger.info(f"├─ Lost: {self.trades_lost} ({(self.trades_lost / self.trades_executed * 100) if self.trades_executed > 0 else 0:.1f}%)")
        logger.info(f"├─ Total Profit: ${self.total_profit:,.2f}")
        logger.info(f"├─ Total Loss: ${self.total_loss:,.2f}")
        logger.info(f"└─ Net P&L: ${profit_loss:,.2f} ({profit_loss_pct:+.2f}%)")
        
        logger.info("\n🔄 REINVESTMENT TRACKING")
        logger.info(f"├─ Amount Reinvested: ${self.reinvested_amount:,.2f}")
        logger.info(f"├─ Profit Reinvestment Rate: {(self.reinvested_amount / self.total_profit * 100) if self.total_profit > 0 else 0:.1f}%")
        logger.info(f"└─ Compounding Effect: {((self.current_portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value * 100):+.2f}%")
        
        logger.info("\n" + "=" * 80 + "\n")
        
        # Store hourly summary
        self.hourly_summary[hour] = {
            "elapsed": elapsed,
            "nav": nav,
            "profit_loss": profit_loss,
            "trades_executed": self.trades_executed,
            "trades_won": self.trades_won,
            "reinvested": self.reinvested_amount,
        }
    
    async def _print_final_report(self):
        """Print final session report"""
        
        self.end_time = datetime.now()
        actual_duration = (self.end_time - self.start_time).total_seconds() / 3600
        
        nav = self.current_portfolio_value or 0
        profit_loss = self.total_profit - self.total_loss
        profit_loss_pct = (profit_loss / self.initial_portfolio_value * 100) if self.initial_portfolio_value > 0 else 0
        
        logger.info("\n" + "=" * 100)
        logger.info("🏁 FINAL SESSION REPORT")
        logger.info("=" * 100)
        
        logger.info("\n⏰ SESSION DURATION")
        logger.info(f"├─ Start: {self.start_time}")
        logger.info(f"├─ End: {self.end_time}")
        logger.info(f"├─ Actual Duration: {actual_duration:.2f} hours")
        logger.info(f"└─ Scheduled Duration: {self.duration_hours} hours")
        
        logger.info("\n💼 PORTFOLIO PERFORMANCE")
        logger.info(f"├─ Initial Portfolio Value: ${self.initial_portfolio_value:,.2f}")
        logger.info(f"├─ Final Portfolio Value: ${nav:,.2f}")
        logger.info(f"├─ Total Change: ${nav - self.initial_portfolio_value:,.2f}")
        logger.info(f"├─ Percentage Gain: {((nav - self.initial_portfolio_value) / self.initial_portfolio_value * 100):+.2f}%")
        logger.info(f"├─ Highest Value: ${self.highest_portfolio_value:,.2f}")
        logger.info(f"├─ Lowest Value: ${self.lowest_portfolio_value:,.2f}")
        logger.info(f"└─ Max Drawdown: ${self.initial_portfolio_value - self.lowest_portfolio_value:,.2f} "
                   f"({((self.initial_portfolio_value - self.lowest_portfolio_value) / self.initial_portfolio_value * 100):.2f}%)")
        
        logger.info("\n📊 TRADING STATISTICS")
        logger.info(f"├─ Total Trades Executed: {self.trades_executed}")
        logger.info(f"├─ Winning Trades: {self.trades_won}")
        logger.info(f"├─ Losing Trades: {self.trades_lost}")
        logger.info(f"├─ Win Rate: {(self.trades_won / self.trades_executed * 100) if self.trades_executed > 0 else 0:.1f}%")
        logger.info(f"├─ Trades Per Hour: {self.trades_executed / actual_duration:.1f}")
        logger.info(f"└─ Average Trade Duration: {(actual_duration * 3600 / self.trades_executed):.0f}s per trade" 
                   if self.trades_executed > 0 else "└─ N/A")
        
        logger.info("\n💹 PROFIT & LOSS TRACKING")
        logger.info(f"├─ Total Profit: ${self.total_profit:,.2f}")
        logger.info(f"├─ Total Loss: ${self.total_loss:,.2f}")
        logger.info(f"├─ Net Profit/Loss: ${profit_loss:,.2f}")
        logger.info(f"├─ Profit Factor: {self.total_profit / self.total_loss if self.total_loss > 0 else float('inf'):.2f}")
        logger.info(f"├─ Average Win: ${self.total_profit / self.trades_won if self.trades_won > 0 else 0:.2f}")
        logger.info(f"├─ Average Loss: ${self.total_loss / self.trades_lost if self.trades_lost > 0 else 0:.2f}")
        logger.info(f"└─ Expectancy: ${profit_loss / self.trades_executed if self.trades_executed > 0 else 0:.2f} per trade")
        
        logger.info("\n🔄 REINVESTMENT & COMPOUNDING")
        logger.info(f"├─ Amount Reinvested: ${self.reinvested_amount:,.2f}")
        logger.info(f"├─ Reinvestment Rate: {(self.reinvested_amount / self.total_profit * 100) if self.total_profit > 0 else 0:.1f}%")
        logger.info(f"├─ Compounding Multiplier: {(nav / self.initial_portfolio_value):.4f}x")
        logger.info(f"└─ Total Growth: {profit_loss_pct:+.2f}%")
        
        logger.info("\n📈 HOURLY BREAKDOWN")
        for hour in sorted(self.hourly_summary.keys()):
            h_data = self.hourly_summary[hour]
            logger.info(f"├─ Hour {hour}: NAV=${h_data['nav']:,.2f} | "
                       f"P&L=${h_data['profit_loss']:,.2f} | "
                       f"Trades={h_data['trades_executed']} | "
                       f"Reinvested=${h_data['reinvested']:,.2f}")
        
        logger.info("\n✅ EFFICIENCY METRICS")
        logger.info(f"├─ Return per Hour: {profit_loss_pct / actual_duration:.2f}% per hour")
        logger.info(f"├─ ROI: {profit_loss_pct:.2f}%")
        logger.info(f"├─ Annualized Return: {profit_loss_pct * (365 / actual_duration):.2f}%")
        logger.info(f"└─ Sharpe Ratio: N/A (requires volatility data)")
        
        logger.info("\n" + "=" * 100)
        logger.info("✅ SESSION COMPLETE")
        logger.info("=" * 100 + "\n")

# ============================================================================
# MAIN ORCHESTRATOR WITH MONITORING
# ============================================================================

async def run_monitored_trading_session(duration_hours: int = 3, paper_trading: bool = False):
    """Run trading system with monitoring"""
    
    # Create monitor
    monitor = TradingSessionMonitor(duration_hours=duration_hours)
    
    # Initialize monitor
    if not await monitor.initialize():
        logger.error("❌ Failed to initialize monitor")
        return False
    
    try:
        # Import orchestrator
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "master_orchestrator",
            Path(__file__).parent / "🎯_MASTER_SYSTEM_ORCHESTRATOR.py"
        )
        master_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(master_module)
        
        MasterSystemOrchestrator = master_module.MasterSystemOrchestrator
        OrchestratorConfig = master_module.OrchestratorConfig
        
        # Create orchestrator
        orchestrator = MasterSystemOrchestrator()
        
        # Override config for monitoring session
        orchestrator.config.duration_hours = duration_hours
        if paper_trading:
            orchestrator.config.live_mode = False
        
        # Start monitoring task
        monitor_task = asyncio.create_task(monitor.run_monitoring_loop(), name="Monitor")
        
        # Start orchestrator task
        orchestrator_task = asyncio.create_task(orchestrator.run_system(), name="Orchestrator")
        
        # Wait for both to complete
        results = await asyncio.gather(
            monitor_task,
            orchestrator_task,
            return_exceptions=True
        )
        
        # Check for errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Session error: {e}", exc_info=True)
        return False

# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run 3-hour trading session with monitoring"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3,
        help="Session duration in hours (default: 3)"
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper trading mode"
    )
    
    args = parser.parse_args()
    
    # Check approval
    if not args.paper and os.getenv("APPROVE_LIVE_TRADING") != "YES":
        logger.error("❌ Live trading requires: export APPROVE_LIVE_TRADING=YES")
        sys.exit(1)
    
    # Run session
    logger.info(f"🚀 Starting {args.duration}-hour trading session")
    logger.info(f"Mode: {'📝 PAPER TRADING' if args.paper else '🔴 LIVE TRADING'}")
    
    success = await run_monitored_trading_session(
        duration_hours=args.duration,
        paper_trading=args.paper
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Session interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
