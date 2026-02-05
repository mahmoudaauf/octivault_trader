import asyncio
from datetime import datetime
from core.utils.structured_logger import get_logger

logger = get_logger("SystemSummary")

async def system_summary_logger(app_context, interval_seconds=60):
    """
    Logs periodic summary of trades, PnL, agent activity, and system status.
    """
    while True:
        try:
            shared_state = app_context.shared_state
            agent_manager = app_context.agent_manager
            pnl_calculator = getattr(app_context, "pnl_calculator", None)
            execution_manager = app_context.execution_manager

            # Timestamp header
            logger.info("\n\n🧠 SYSTEM SUMMARY [{}]".format(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))

            # === Trade + PnL ===
            trades = shared_state.get_trade_history()
            active = shared_state.get_active_trades()
            total_trades = len(trades)
            active_count = len(active)
            roi = shared_state.get_roi()
            total_pnl = shared_state.get_total_pnl()

            logger.info(f"📊 Trades: {total_trades} | Active: {active_count} | PnL: {total_pnl:.2f} | ROI: {roi:.2f}%")

            # === Agent Activity ===
            agent_signals = shared_state.get_latest_agent_signals()
            if agent_signals:
                logger.info("🤖 Agent Signals (Last):")
                for agent, signals in agent_signals.items():
                    logger.info(f"- {agent}: {len(signals)} signals")
            else:
                logger.info("🤖 Agent Signals: None recorded.")

            # === Execution Health ===
            last_executions = shared_state.get_recent_executions(limit=5)
            logger.info("🚀 Recent Executions:")
            if last_executions:
                for ex in last_executions:
                    logger.info(f"- {ex['symbol']} | {ex['action']} | filled={ex['filled']} | price={ex['price']} | time={ex['timestamp']}")
            else:
                logger.info("- No executions logged.")

        except Exception as e:
            logger.error(f"🔥 Error during system summary: {e}", exc_info=True)

        await asyncio.sleep(interval_seconds)
