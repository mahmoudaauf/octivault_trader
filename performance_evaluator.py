import asyncio
import csv
import os
from datetime import datetime
import numpy as np
import logging
from core.health import update_health
from core.component_status_logger import ComponentStatusLogger

# Ensure logging is configured for this module if it's not handled globally
logger = logging.getLogger("PerformanceEvaluator") # Changed name to match class
# Assuming global logging config handles handlers, if not, add basic setup here
# if not logger.handlers:
#     ch = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

class PerformanceEvaluator: # <--- THIS IS THE CRUCIAL CHANGE!
    def __init__(self, shared_state, config, alert_callback=None):
        self.shared_state = shared_state
        self.config = config
        self.alert_callback = alert_callback

        # Log level from config, defaulting to INFO if not found
        log_level = getattr(config, "PERFORMANCE_LOG_LEVEL", "INFO").upper()
        # Using the instance logger for consistency within the class
        self.logger = logging.getLogger("PerformanceEvaluator") # Ensure this is also an instance logger
        self.logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        # This part of logging setup should ideally be handled by a global configure_logging
        # If it's still needed per-module, keep it, but usually not for handlers.
        # if not self.logger.hasHandlers():
        #     ch = logging.StreamHandler()
        #     ch.setLevel(getattr(logging, log_level, logging.INFO))
        #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #     ch.setFormatter(formatter)
        #     self.logger.addHandler(ch)

        self.performance_log = "logs/performance.csv"
        os.makedirs("logs", exist_ok=True)
        if not os.path.exists(self.performance_log):
            with open(self.performance_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "total_pnl", "pnl_hour", "roi_avg", "sharpe_avg",
                    "drawdown_avg", "win_rate_avg", "trades_per_hour",
                    "ipo_total_pnl", "ipo_win_rate", "ipo_roi"
                ])

        self.last_check = datetime.utcnow()
        self.logger.info("PerformanceEvaluator initialized.") # Using self.logger


    async def track_performance(self):
        """
        Placeholder for background task to track performance metrics.
        You will need to implement the actual logic here.
        This method will likely run periodically to aggregate data.
        """
        self.logger.info("PerformanceEvaluator: Starting background performance tracking.")
        while True:
            try:
                # Example: log current PnL, calculate other metrics periodically
                # This could fetch data from shared_state.metrics
                current_pnl = self.shared_state.metrics.get("total_pnl", 0.0)
                current_roi = self.shared_state.metrics.get("roi", 0.0)
                
                # In a real implementation, you'd calculate more complex metrics like:
                # - Daily PnL
                # - Sharpe Ratio over a period
                # - Max Drawdown
                # - Win Rate
                # - Trades per hour
                # - IPO-specific metrics (if agents provide them)

                timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.performance_log, "a", newline="") as f:
                    writer = csv.writer(f)
                    # These values are placeholders until actual calculations are implemented
                    writer.writerow([
                        timestamp, f"{current_pnl:.2f}", "N/A", f"{current_roi:.4f}", "N/A",
                        "N/A", "N/A", "N/A",
                        "N/A", "N/A", "N/A"
                    ])
                self.logger.debug(f"Performance logged: Total PnL={current_pnl:.2f}, ROI={current_roi:.2%}")

            except Exception as e:
                self.logger.exception(f"🚨 Error in background performance tracking: {e}")
            
            # This interval should be independent of KPI evaluation if track_performance
            # is meant for frequent data logging, while evaluate_kpis is for alerts/summary.
            # Use getattr for robustness in case config doesn't have the attribute
            await asyncio.sleep(getattr(self.config, "PERFORMANCE_TRACKING_INTERVAL", 300)) # Use a specific config for this, default to 300s (5min)


    async def evaluate_performance(self): # Renamed from evaluate_kpis
        """
        Placeholder for evaluating Key Performance Indicators.
        This method should check current metrics against thresholds and trigger alerts.
        You will need to implement the actual KPI evaluation logic here.
        """
        self.logger.debug("PerformanceEvaluator: Evaluating KPIs.")
        # Example: Check if daily PnL exceeds a target or falls below a threshold
        target_per_hour = getattr(self.config, "BASE_TARGET_PER_HOUR", 0.0) # Use getattr for robustness
        
        current_total_pnl = self.shared_state.metrics.get("total_pnl", 0.0)
        current_roi = self.shared_state.metrics.get("roi", 0.0)

        # This is a very simplified check; real KPIs involve historical data, volatility, etc.
        # current_total_pnl = self.shared_state.metrics.get("total_pnl", 0.0)
        # current_roi = self.shared_state.metrics.get("roi", 0.0)

        # Example KPI check: if ROI is negative beyond a certain point, trigger alert
        if current_roi < -0.02: # If ROI drops below -2%
            self.logger.warning(f"📉 Performance alert: ROI is {current_roi:.2%}. Review strategy.")
            if self.alert_callback:
                # Assuming alert_callback is a function that sends a message
                await self.alert_callback("PERFORMANCE_ALERT", {"message": f"ROI dropped to {current_roi:.2%}", "severity": "warning"})
        
        # Example: Check if current total PnL is meeting some hourly target
        # This requires tracking PnL over time, not just current snapshot.
        # This method also needs to be robustly implemented.
        
        # Update system health - this might be handled by the run loop's outer status update
        # from core.component_status_logger import ComponentStatusLogger # Local import to avoid circular dependency
        # await ComponentStatusLogger.log_status(
        #     "PerformanceEvaluator", "Operational",
        #     f"KPIs evaluated. Total PnL: {current_total_pnl:.2f}, ROI: {current_roi:.2%}"
        # )


    async def report_health_loop(self):
        while True:
            ComponentStatusLogger.log_status(
                component_name="PerformanceEvaluator",
                status="Healthy",
                detail=f"PerformanceEvaluator KPI loop heartbeat at {datetime.now().isoformat()}"
            )
            await asyncio.sleep(getattr(self.config, "PERFORMANCE_HEALTH_INTERVAL", 30))

    async def _main_loop(self, interval):
        while True:
            try:
                await self.evaluate_performance()
                # Restore the original status update logic for the main loop
                await self.shared_state.update_component_status("PerformanceEvaluator", "Operational")
            except Exception as e:
                self.logger.exception("❌ PerformanceEvaluator encountered an error.")
                await self.shared_state.update_component_status("PerformanceEvaluator", "Error")
            await asyncio.sleep(interval)


    async def run(self):
        await update_health(self.shared_state, "PerformanceEvaluator", "Healthy", "Evaluating KPIs and strategy metrics.")
        self.logger.info("✅ PerformanceEvaluator is running.")
        ComponentStatusLogger.log_status(
            component_name="PerformanceEvaluator",
            status="Healthy",
            detail="PerformanceEvaluator started and reporting health."
        )

        interval = getattr(self.config, "PERFORMANCE_EVALUATION_INTERVAL", 60)

        await asyncio.gather(
            self._main_loop(interval),
            self.report_health_loop()
        )
        
    async def run_loop(self):
        """Wrapper for Phase 9 compatibility."""
        await self.run()
