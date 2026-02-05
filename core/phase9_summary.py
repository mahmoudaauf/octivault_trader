# core/phase9_summary.py
import logging

logger = logging.getLogger("Phase9Summary")

def get_component_status(name, instance):
    if instance is None:
        return f"❌ {name}: Not initialized."
    elif hasattr(instance, "run") or hasattr(instance, "run_loop"):
        return f"✅ {name}: Initialized and runnable."
    else:
        return f"⚠️ {name}: Initialized but no run() or run_loop() method."

async def log_phase9_summary(app_context):
    logger.info("\n📋 PHASE 9 COMPONENT LAUNCH SUMMARY")

    components = {
        "MetaController": app_context.meta_controller,
        "ExecutionManager": app_context.execution_manager,
        "TPSLEngine": app_context.tp_sl_engine,
        "RiskManager": app_context.risk_manager,
        "PortfolioManager": app_context.portfolio_manager,
        "CompoundingEngine": app_context.compounding_engine,
        "PerformanceMonitor": app_context.performance_monitor,
        "PositionManager": app_context.position_manager,
        "AgentManager": app_context.agent_manager,
        "PerformanceEvaluator": getattr(app_context, "performance_evaluator", None),
        "PnLCalculator": getattr(app_context, "pnl_calculator", None),
        "Watchdog": app_context.watchdog,
        "Heartbeat": app_context.heartbeat,
        "StrategyManager": app_context.strategy_manager,
    }

    for name, instance in components.items():
        status = get_component_status(name, instance)
        logger.info(status)

# Optional: health reporting helpers
async def report_health_meta_controller(shared_state):
    from core.health import update_health
    await update_health(shared_state, "MetaController", "Healthy", "Aggregating signals.")

async def report_health_agent_manager(shared_state):
    from core.health import update_health
    await update_health(shared_state, "AgentManager", "Healthy", "Running strategy agent loops.")

async def report_health_performance_evaluator(shared_state):
    from core.health import update_health
    await update_health(shared_state, "PerformanceEvaluator", "Healthy", "Evaluating KPIs and PnL trends.")

async def report_health_pnl_calculator(shared_state):
    from core.health import update_health
    await update_health(shared_state, "PnLCalculator", "Healthy", "Tracking real-time profit and loss.")
