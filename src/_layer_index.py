"""Single source of truth for layer → module assignments.

Keys are the short names exposed inside `src.lN_*` packages.
Values are the canonical dotted module paths after Phase C migration.
"""
from __future__ import annotations
from typing import Dict

LAYER_MODULES: Dict[str, Dict[str, str]] = {
    # L0 — Cross-cutting (Phase C-L0 ✅ for core/, utils/ kept in place) -----
    "l0_core": {
        "contracts":             "src.l0_core.contracts",
        "config":                "src.l0_core.config",
        "config_constants":      "src.l0_core.config_constants",
        "config_validator":      "src.l0_core.config_validator",
        "error_types":           "src.l0_core.error_types",
        "error_handler":         "src.l0_core.error_handler",
        "core_utils":            "src.l0_core.core_utils",
        "logger_utils":          "src.l0_core.logger_utils",
        "stubs":                 "src.l0_core.stubs",
        "layer_contracts":       "src.l0_core.layer_contracts",
        "shared_state":          "src.l0_core.shared_state",
        "component_status_logger": "src.l0_core.component_status_logger",
        "health":                "src.l0_core.health",
        "healthy":               "src.l0_core.healthy",
        "metrics":               "src.l0_core.metrics",
        "time_utils":            "src.l0_core.time_utils",
        "indicators":            "utils.indicators",
        "ta_indicators":         "utils.ta_indicators",
        "hyg_guards":            "utils.hyg_guards",
        "tuned_params":          "utils.tuned_params",
        "pnl_calculator":        "utils.pnl_calculator",
        "logging_setup":         "utils.logging_setup",
        "balance_threshold_config": "balance_threshold_config",
    },

    # L1 — Exchange I/O (Phase C-L1 ✅) -------------------------------------
    "l1_exchange": {
        "exchange_client":        "src.l1_exchange.exchange_client",
        "exchange_truth_auditor": "src.l1_exchange.exchange_truth_auditor",
        "order_cache_manager":    "src.l1_exchange.order_cache_manager",
        "ws_market_data":         "src.l1_exchange.ws_market_data",
        "market_data_websocket":  "src.l1_exchange.market_data_websocket",
        "polling_coordinator":    "src.l1_exchange.polling_coordinator",
        "balance_sync_backoff":   "src.l1_exchange.balance_sync_backoff",
        "retry_manager":          "src.l1_exchange.retry_manager",
    },

    # L2 — Wallet & Market data (Phase C-L2 ✅) -----------------------------
    "l2_marketdata": {
        "balance_manager":           "src.l2_marketdata.balance_manager",
        "market_data_feed":          "src.l2_marketdata.market_data_feed",
        "heartbeat":                 "src.l2_marketdata.heartbeat",
        "correlation_manager":       "src.l2_marketdata.correlation_manager",
        "market_regime_detector":    "src.l2_marketdata.market_regime_detector",
        "market_regime_integration": "src.l2_marketdata.market_regime_integration",
        "volatility_regime":         "src.l2_marketdata.volatility_regime",
        "nav_regime":                "src.l2_marketdata.nav_regime",
        "regime_proposal_analyzer":  "src.l2_marketdata.regime_proposal_analyzer",
        "anomaly_detection":         "src.l2_marketdata.anomaly_detection",
    },

    # L3 — Portfolio & state (Phase C-L3 ✅) --------------------------------
    "l3_portfolio": {
        "portfolio_authority":          "src.l3_portfolio.portfolio_authority",
        "portfolio_manager":            "src.l3_portfolio.portfolio_manager",
        "portfolio_balancer":           "src.l3_portfolio.portfolio_balancer",
        "portfolio_buckets":            "src.l3_portfolio.portfolio_buckets",
        "portfolio_segmentation":       "src.l3_portfolio.portfolio_segmentation",
        "three_bucket_manager":         "src.l3_portfolio.three_bucket_manager",
        "bucket_classifier":            "src.l3_portfolio.bucket_classifier",
        "position_manager":             "src.l3_portfolio.position_manager",
        "position_merger_enhanced":     "src.l3_portfolio.position_merger_enhanced",
        "position_operation_validator": "src.l3_portfolio.position_operation_validator",
        "restart_position_classifier":  "src.l3_portfolio.restart_position_classifier",
        "holding_utility":              "src.l3_portfolio.holding_utility",
        "symbol_manager":               "src.l3_portfolio.symbol_manager",
        "symbol_rotation":              "src.l3_portfolio.symbol_rotation",
        "rotation_authority":           "src.l3_portfolio.rotation_authority",
        "universe_rotation_engine":     "src.l3_portfolio.universe_rotation_engine",
        "bootstrap_symbols":            "src.l3_portfolio.bootstrap_symbols",
        "bootstrap_manager":            "src.l3_portfolio.bootstrap_manager",
        "discovery_coordinator":        "src.l3_portfolio.discovery_coordinator",
        "trade_journal":                "src.l3_portfolio.trade_journal",
        "event_store":                  "src.l3_portfolio.event_store",
        "state_manager":                "src.l3_portfolio.state_manager",
        "state_synchronizer":           "src.l3_portfolio.state_synchronizer",
        "replay_engine":                "src.l3_portfolio.replay_engine",
        "dead_capital_healer":          "src.l3_portfolio.dead_capital_healer",
        "reserve_manager":              "src.l3_portfolio.reserve_manager",
        "system_state_manager":         "system_state_manager",
    },

    # L4 — Execution & Order Mgmt (Phase C-L4 ✅) ---------------------------
    "l4_execution": {
        "execution_manager":       "src.l4_execution.execution_manager",
        "execution_logic":         "src.l4_execution.execution_logic",
        "maker_execution":         "src.l4_execution.maker_execution",
        "action_router":           "src.l4_execution.action_router",
        "cash_router":             "src.l4_execution.cash_router",
        "intent_manager":          "src.l4_execution.intent_manager",
        "tp_sl_engine":            "src.l4_execution.tp_sl_engine",
        "exit_arbitrator":         "src.l4_execution.exit_arbitrator",
        "exit_utils":              "src.l4_execution.exit_utils",
        "profit_target_engine":    "src.l4_execution.profit_target_engine",
        "liquidation_orchestrator":"src.l4_execution.liquidation_orchestrator",
        "leverage_manager":        "src.l4_execution.leverage_manager",
        "signal_batcher":          "src.l4_execution.signal_batcher",
        "trading_coordinator":     "src.l4_execution.trading_coordinator",
        "trading_hours_manager":   "src.l4_execution.trading_hours_manager",
        "recovery_engine":         "src.l4_execution.recovery_engine",
        "auto_recovery":           "src.l8_lifecycle.runners.auto_recovery",
        "apply_recovery_to_live":  "src.l8_lifecycle.runners.apply_recovery_to_live",
    },

    # L5 — Strategy & Decision (Phase C-L5 ✅) ------------------------------
    "l5_strategy": {
        "dip_sniper":          "agents.dip_sniper",
        "edge_calculator":     "agents.edge_calculator",
        "ipo_chaser":          "agents.ipo_chaser",
        "liquidation_agent":   "agents.liquidation_agent",
        "ml_forecaster":       "agents.ml_forecaster",
        "swing_trade_hunter":  "agents.swing_trade_hunter",
        "symbol_screener":     "agents.symbol_screener",
        "trend_hunter":        "agents.trend_hunter",
        "wallet_scanner_agent":"agents.wallet_scanner_agent",
        "signal_manager":              "src.l5_strategy.signal_manager",
        "signal_fusion":               "src.l5_strategy.signal_fusion",
        "arbitration_engine":          "src.l5_strategy.arbitration_engine",
        "opportunity_ranker":          "src.l5_strategy.opportunity_ranker",
        "baseline_trading_kernel":     "src.l5_strategy.baseline_trading_kernel",
        "agent_manager":               "src.l5_strategy.agent_manager",
        "agent_optimizer":             "src.l5_strategy.agent_optimizer",
        "agent_registry":              "src.l5_strategy.agent_registry",
        "external_adoption_engine":    "src.l5_strategy.external_adoption_engine",
        "objective_feedback_controller":"src.l5_strategy.objective_feedback_controller",
        "performance_evaluator":       "src.l5_strategy.performance_evaluator",
        "focus_mode":                  "src.l5_strategy.focus_mode",
        "mode_manager":                "src.l5_strategy.mode_manager",
        "model_manager":               "src.l5_strategy.model_manager",
        "model_trainer":               "src.l5_strategy.model_trainer",
        "capital_velocity_optimizer":  "src.l5_strategy.capital_velocity_optimizer",
        "objective_tracker":           "src.l8_lifecycle.runners.objective_tracker",
    },

    # L6 — Governance & Policy (Phase C-L6 ✅) ------------------------------
    "l6_governance": {
        "risk_manager":            "src.l6_governance.risk_manager",
        "capital_governor":        "src.l6_governance.capital_governor",
        "capital_symbol_governor": "src.l6_governance.capital_symbol_governor",
        "capital_allocator":       "src.l6_governance.capital_allocator",
        "adaptive_capital_engine": "src.l6_governance.adaptive_capital_engine",
        "compounding_engine":      "src.l6_governance.compounding_engine",
        "rebalancing_engine":      "src.l6_governance.rebalancing_engine",
        "policy_manager":          "src.l6_governance.policy_manager",
        "scaling":                 "src.l6_governance.scaling",
        "auto_rule_proposer":      "automation.auto_rule_proposer",
        "proposal_monitor":        "automation.proposal_monitor",
        "rule_overrides":          "automation.rule_overrides",
    },

    # L7 — Observability & UX (Phase C-L7 ✅) -------------------------------
    "l7_observability": {
        "prometheus_exporter":   "src.l7_observability.prometheus_exporter",
        "health_check":          "src.l7_observability.health_check",
        "health_check_manager":  "src.l7_observability.health_check_manager",
        "health_endpoints":      "src.l7_observability.health_endpoints",
        "health_monitor":        "src.l7_observability.health_monitor",
        "alert_system":          "src.l7_observability.alert_system",
        "performance_monitor":   "src.l7_observability.performance_monitor",
        "dashboard":             "src.l7_observability.dashboard",
        "apm_instrument":        "src.l7_observability.apm_instrument",
        "jaeger_tracer":         "src.l7_observability.jaeger_tracer",
        "balance_dashboard":     "src.l7_observability.monitors.balance_dashboard",
        "error_monitor":         "src.l7_observability.monitors.error_monitor",
    },

    # L8 — Lifecycle & Recovery (Phase C-L8 ✅) -----------------------------
    "l8_lifecycle": {
        "lifecycle_manager":     "src.l8_lifecycle.lifecycle_manager",
        "startup_orchestrator":  "src.l8_lifecycle.startup_orchestrator",
        "watchdog":              "src.l8_lifecycle.watchdog",
        "chaos_monkey":          "src.l8_lifecycle.chaos_monkey",
        "meta_controller":       "src.l8_lifecycle.meta_controller",
        "live_integration":      "src.l8_lifecycle.runners.live_integration",
        "component_validator":   "src.l8_lifecycle.runners.component_validator",
    },
}


def layer_of(short_name: str) -> str:
    for layer, mods in LAYER_MODULES.items():
        if short_name in mods:
            return layer
    return ""


__all__ = ["LAYER_MODULES", "layer_of"]
