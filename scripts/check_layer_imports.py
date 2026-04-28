#!/usr/bin/env python3
"""
scripts/check_layer_imports.py
==============================

CI guard for the 8-layer logical architecture defined in
LOGICAL_LAYERED_ARCHITECTURE.md.

Walks every production .py file in the workspace, statically parses its
imports, looks up the layer of the importer and the layer of every
imported workspace module via FILE_LAYER_MAP, and verifies the arrow is
permitted by ALLOWED_DEPENDENCIES (core.layer_contracts).

Exit code:
    0  -> no violations
    1  -> at least one forbidden import (printed)
    2  -> usage error
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.l0_core.layer_contracts import ALLOWED_DEPENDENCIES  # noqa: E402

# ---------------------------------------------------------------------------
# FILE -> LAYER registry. Authoritative mapping of every module to its layer.
# Keep in sync with LOGICAL_LAYERED_ARCHITECTURE.md §3–§11.
# Keys are paths relative to repo root, with forward slashes.
# ---------------------------------------------------------------------------

FILE_LAYER_MAP: Dict[str, str] = {

    # ---- L0: Cross-cutting -------------------------------------------------
    # Ambient/cross-cutting infra used by every layer (logs, metrics primitives,
    # shared blackboard, health markers) — registered at L0 with the network
    # exporters/dashboards remaining at L7.
    "utils/__init__.py":                        "L0",
    "deployment/__init__.py":                   "L0",
    # Phase A namespace packages — re-export only, no business logic:
    "src/__init__.py":                          "L0",
    "src/_layer_index.py":                      "L0",
    "src/_lazy.py":                             "L0",
    "src/l0_core/__init__.py":                  "L0",
    "src/l1_exchange/__init__.py":              "L1",
    "src/l2_marketdata/__init__.py":            "L2",
    "src/l3_portfolio/__init__.py":             "L3",
    "src/l4_execution/__init__.py":             "L4",
    "src/l5_strategy/__init__.py":              "L5",
    "src/l6_governance/__init__.py":            "L6",
    "src/l7_observability/__init__.py":         "L7",
    "src/l7_observability/monitors/__init__.py":"L7",
    "src/l8_lifecycle/__init__.py":             "L8",
    "src/l8_lifecycle/runners/__init__.py":     "L8",
    "utils/logging_setup.py":                   "L0",
    "utils/indicators.py":                      "L0",
    "utils/ta_indicators.py":                   "L0",
    "utils/hyg_guards.py":                      "L0",
    "utils/tuned_params.py":                    "L0",
    "utils/pnl_calculator.py":                  "L0",
    "utils/volatility_adjusted_confidence.py":  "L0",
    "utils/symbol_filter_pipeline.py":          "L0",
    "utils/shared_state_tools.py":              "L0",
    "utils/ohlcv_cache.py":                     "L0",
    "config/__init__.py":                       "L0",
    "config/EV_ALIGNMENT_CONFIG.py":            "L0",
    "balance_threshold_config.py":              "L0",

    # ---- L1: Exchange I/O --------------------------------------------------
    # Canonical (new home after Phase C-L1):
    "src/l1_exchange/exchange_client.py":        "L1",
    "src/l1_exchange/exchange_truth_auditor.py": "L1",
    "src/l1_exchange/order_cache_manager.py":    "L1",
    "src/l1_exchange/ws_market_data.py":         "L1",
    "src/l1_exchange/market_data_websocket.py":  "L1",
    "src/l1_exchange/polling_coordinator.py":    "L1",
    "src/l1_exchange/balance_sync_backoff.py":   "L1",
    "src/l1_exchange/retry_manager.py":          "L1",
    # Backward-compat shims (still classified L1):

    # ---- L2: Wallet & Market data -----------------------------------------
    "stream/__init__.py":                       "L2",

    # ---- L3: Portfolio & state --------------------------------------------
    "portfolio/__init__.py":                    "L3",
    "system_state_manager.py":                  "L3",

    # ---- L4: Execution & Order Mgmt ---------------------------------------
    "tools/recover_missing_sells.py":           "L4",
    "tools/exit_metrics.py":                    "L4",
    "tools/compound_engine.py":                 "L4",
    "src/l8_lifecycle/runners/auto_recovery.py":          "L4",
    "src/l8_lifecycle/runners/apply_recovery_to_live.py": "L4",
    # Backward-compat shims at root (Phase B):
    "auto_recovery.py":                         "L4",
    "apply_recovery_to_live.py":                "L4",

    # ---- L5: Strategy & Decision ------------------------------------------
    "agents/__init__.py":                       "L5",
    "agents/dip_sniper.py":                     "L5",
    "agents/edge_calculator.py":                "L5",
    "agents/ipo_chaser.py":                     "L5",
    "agents/liquidation_agent.py":              "L5",
    "agents/ml_forecaster.py":                  "L5",
    "agents/swing_trade_hunter.py":             "L5",
    "agents/symbol_screener.py":                "L5",
    "agents/trend_hunter.py":                   "L5",
    "agents/wallet_scanner_agent.py":           "L5",
    "diagnostic_signal_flow.py":                "L5",
    "SIGNAL_FLOW_DIAGNOSTIC.py":                "L5",

    # ---- L6: Governance & Policy ------------------------------------------
    "automation/auto_rule_proposer.py":         "L6",
    "automation/proposal_monitor.py":           "L6",
    "automation/rule_overrides.py":             "L6",

    # ---- L7: Observability & UX -------------------------------------------
    "dashboards/__init__.py":                   "L7",
    "diagnostics/per_loop_symbol_diag.py":      "L7",
    "core/diagnostics/system_summary.py":       "L7",
    "src/l7_observability/diagnostics/__init__.py":      "L7",
    "src/l7_observability/diagnostics/system_summary.py": "L7",
    "tools/next_level_tpsl_analysis.py":        "L7",
    "tools/monitor_6h_session.py":              "L7",
    "monitoring/sandbox_monitor.py":            "L7",
    "src/l7_observability/monitors/balance_dashboard.py":         "L7",
    "src/l7_observability/monitors/error_monitor.py":             "L7",
    "src/l7_observability/monitors/extract_rejections.py":        "L7",
    "src/l7_observability/monitors/phase2_monitoring.py":         "L7",
    "src/l7_observability/monitors/monitor_phase2_realtime.py":   "L7",
    "src/l7_observability/monitors/monitor_4hour_session.py":     "L7",
    "ANALYSIS_REPORT.py":                       "L7",
    "FAST_DIAGNOSTICS.py":                      "L7",
    "LIVE_MONITOR.py":                          "L7",
    "LIVE_PHASE2_MONITOR.py":                   "L7",
    "LIVE_TRADING_WITH_BALANCE_MONITOR.py":     "L7",
    "MONITOR_15MIN_REALTIME.py":                "L7",
    "MONITOR_15MIN_SESSION.py":                 "L7",
    "MONITOR_3HOUR_TRADING_SESSION.py":         "L7",
    "PERIODIC_MONITOR.py":                      "L7",
    "REALTIME_15MIN_MONITOR.py":                "L7",
    "REALTIME_DIAGNOSTICS.py":                  "L7",
    "REALTIME_MONITOR.py":                      "L7",
    "REALTIME_SESSION_MONITOR.py":              "L7",
    "CONTINUOUS_ACTIVE_MONITOR.py":             "L7",
    "CONTINUOUS_MONITOR.py":                    "L7",
    "6HOUR_MONITORING_DASHBOARD.py":            "L7",
    "PROFIT_ACCUMULATOR_MONITOR.py":            "L7",
    "PHASE_2_STATUS_REPORT.py":                 "L7",
    "SESSION_STATUS_REPORT.py":                 "L7",

    # ---- L8: Lifecycle & Recovery -----------------------------------------
    "tools/diagnose_runtime.py":                "L8",
    "FORCE_SIGNALS_INJECTOR.py":                "L8",
    "🎯_MASTER_SYSTEM_ORCHESTRATOR.py":         "L8",
    "AUTONOMOUS_STARTUP_GUIDE.py":              "L8",
    "AUTONOMOUS_SYSTEM_STARTUP.py":             "L8",
    "RUN_AUTONOMOUS_LIVE.py":                   "L8",
    "RUN_3HOUR_SESSION.py":                     "L8",
    "run_4hour_session.py":                     "L8",
    "RUN_6HOUR_SESSION.py":                     "L8",
    "RUN_6HOUR_SESSION_MONITORED.py":           "L8",
    "2HOUR_CHECKPOINT_SESSION.py":              "L8",
    "PRODUCTION_STARTUP.py":                    "L8",
    "PERSISTENT_TRADING_WATCHDOG.py":           "L8",
    "GATING_WATCHDOG.py":                       "L8",
    "phase2_paper_trading.py":                  "L8",
    "phase3_live_trading.py":                   "L8",
    "phase4_30min_test.py":                     "L8",
    "phase4_quick_validation.py":               "L8",
    "phase4_verify.py":                         "L8",
    "deploy_phase2_production.py":              "L8",
    "src/l8_lifecycle/runners/verify_deployment.py":      "L8",
    "src/l8_lifecycle/runners/verify_dust_fix.py":        "L8",
    "src/l8_lifecycle/runners/verify_fixes.py":           "L8",
    "src/l8_lifecycle/runners/verify_fixes_detailed.py":  "L8",
    "src/l8_lifecycle/runners/live_integration.py":       "L8",
    "src/l8_lifecycle/runners/component_validator.py":    "L8",
    "src/l8_lifecycle/runners/objective_tracker.py":      "L5",
    # Backward-compat shim:
    "live_integration.py":                      "L8",
    "CONTINUOUS_OPERATION_GUIDE.py":            "L8",
    "TEST_BOOTSTRAP.py":                        "L8",
    "TEST_EXIT_FIRST_VALIDATION.py":            "L8",
    "TEST_FALLBACK.py":                         "L8",
    "test_rounding_fix.py":                     "L8",
    "test_trendhunter_import.py":               "L8",
    "UNIT_TEST_EXECUTION_GUIDE.py":             "L8",
    "tools/__init__.py":                        "L8",
    "tools/fix_indentation.py":                 "L8",
    "tools/fix_python_indentation.py":          "L8",
    "tools/advanced_fix_python_indentation.py": "L8",
    "tools/fix_class_decorator_indentation.py": "L8",
    "tools/smart_python_indentation_fixer.py":  "L8",
    "scripts/__init__.py":                      "L8",
    "scripts/type_check_analyzer.py":           "L8",
    "scripts/check_layer_imports.py":           "L8",
    "scripts/migrate_to_layer.py":              "L8",
    "scripts/migrate_consumers.py":             "L8",

    # ==== Phase C-L0 canonical paths ====
    # ---- L0: l0_core ----
    "src/l0_core/__init__.py":                  "L0",
    "src/l0_core/contracts.py":                 "L0",
    "src/l0_core/config.py":                    "L0",
    "src/l0_core/config_constants.py":          "L0",
    "src/l0_core/config_validator.py":          "L0",
    "src/l0_core/error_types.py":               "L0",
    "src/l0_core/error_handler.py":             "L0",
    "src/l0_core/core_utils.py":                "L0",
    "src/l0_core/logger_utils.py":              "L0",
    "src/l0_core/stubs.py":                     "L0",
    "src/l0_core/layer_contracts.py":           "L0",
    "src/l0_core/shared_state.py":              "L0",
    "src/l0_core/component_status_logger.py":   "L0",
    "src/l0_core/health.py":                    "L0",
    "src/l0_core/healthy.py":                   "L0",
    "src/l0_core/metrics.py":                   "L0",
    "src/l0_core/time_utils.py":                "L0",

    # ==== Phase C-L2..L8 canonical paths (auto-generated from src/lN_*/) ====
    # ---- L2: l2_marketdata ----
    "src/l2_marketdata/anomaly_detection.py":                    "L2",
    "src/l2_marketdata/balance_manager.py":                      "L2",
    "src/l2_marketdata/correlation_manager.py":                  "L2",
    "src/l2_marketdata/heartbeat.py":                            "L2",
    "src/l2_marketdata/market_data_feed.py":                     "L2",
    "src/l2_marketdata/market_regime_detector.py":               "L2",
    "src/l2_marketdata/market_regime_integration.py":            "L2",
    "src/l2_marketdata/nav_regime.py":                           "L2",
    "src/l2_marketdata/regime_proposal_analyzer.py":             "L2",
    "src/l2_marketdata/volatility_regime.py":                    "L2",
    # ---- L3: l3_portfolio ----
    "src/l3_portfolio/bootstrap_manager.py":                     "L3",
    "src/l3_portfolio/bootstrap_symbols.py":                     "L3",
    "src/l3_portfolio/bucket_classifier.py":                     "L3",
    "src/l3_portfolio/dead_capital_healer.py":                   "L3",
    "src/l3_portfolio/discovery_coordinator.py":                 "L3",
    "src/l3_portfolio/event_store.py":                           "L3",
    "src/l3_portfolio/holding_utility.py":                       "L3",
    "src/l3_portfolio/portfolio_authority.py":                   "L3",
    "src/l3_portfolio/portfolio_balancer.py":                    "L3",
    "src/l3_portfolio/portfolio_buckets.py":                     "L3",
    "src/l3_portfolio/portfolio_manager.py":                     "L3",
    "src/l3_portfolio/portfolio_segmentation.py":                "L3",
    "src/l3_portfolio/position_manager.py":                      "L3",
    "src/l3_portfolio/position_merger_enhanced.py":              "L3",
    "src/l3_portfolio/position_operation_validator.py":          "L3",
    "src/l3_portfolio/replay_engine.py":                         "L3",
    "src/l3_portfolio/reserve_manager.py":                       "L3",
    "src/l3_portfolio/restart_position_classifier.py":           "L3",
    "src/l3_portfolio/rotation_authority.py":                    "L3",
    "src/l3_portfolio/state_manager.py":                         "L3",
    "src/l3_portfolio/state_synchronizer.py":                    "L3",
    "src/l3_portfolio/symbol_manager.py":                        "L3",
    "src/l3_portfolio/symbol_rotation.py":                       "L3",
    "src/l3_portfolio/three_bucket_manager.py":                  "L3",
    "src/l3_portfolio/trade_journal.py":                         "L3",
    "src/l3_portfolio/universe_rotation_engine.py":              "L3",
    # ---- L4: l4_execution ----
    "src/l4_execution/action_router.py":                         "L4",
    "src/l4_execution/cash_router.py":                           "L4",
    "src/l4_execution/execution_logic.py":                       "L4",
    "src/l4_execution/execution_manager.py":                     "L4",
    "src/l4_execution/exit_arbitrator.py":                       "L4",
    "src/l4_execution/exit_utils.py":                            "L4",
    "src/l4_execution/intent_manager.py":                        "L4",
    "src/l4_execution/leverage_manager.py":                      "L4",
    "src/l4_execution/liquidation_orchestrator.py":              "L4",
    "src/l4_execution/maker_execution.py":                       "L4",
    "src/l4_execution/profit_target_engine.py":                  "L4",
    "src/l4_execution/recovery_engine.py":                       "L4",
    "src/l4_execution/signal_batcher.py":                        "L4",
    "src/l4_execution/tp_sl_engine.py":                          "L4",
    "src/l4_execution/trading_coordinator.py":                   "L4",
    "src/l4_execution/trading_hours_manager.py":                 "L4",
    # ---- L5: l5_strategy ----
    "src/l5_strategy/agent_manager.py":                          "L5",
    "src/l5_strategy/agent_optimizer.py":                        "L5",
    "src/l5_strategy/agent_registry.py":                         "L5",
    "src/l5_strategy/arbitration_engine.py":                     "L5",
    "src/l5_strategy/baseline_trading_kernel.py":                "L5",
    "src/l5_strategy/capital_velocity_optimizer.py":             "L5",
    "src/l5_strategy/external_adoption_engine.py":               "L5",
    "src/l5_strategy/focus_mode.py":                             "L5",
    "src/l5_strategy/mode_manager.py":                           "L5",
    "src/l5_strategy/model_manager.py":                          "L5",
    "src/l5_strategy/model_trainer.py":                          "L5",
    "src/l5_strategy/objective_feedback_controller.py":          "L5",
    "src/l5_strategy/opportunity_ranker.py":                     "L5",
    "src/l5_strategy/performance_evaluator.py":                  "L5",
    "src/l5_strategy/signal_fusion.py":                          "L5",
    "src/l5_strategy/signal_manager.py":                         "L5",
    # ---- L6: l6_governance ----
    "src/l6_governance/adaptive_capital_engine.py":              "L6",
    "src/l6_governance/capital_allocator.py":                    "L6",
    "src/l6_governance/capital_governor.py":                     "L6",
    "src/l6_governance/capital_symbol_governor.py":              "L6",
    "src/l6_governance/compounding_engine.py":                   "L6",
    "src/l6_governance/policy_manager.py":                       "L6",
    "src/l6_governance/rebalancing_engine.py":                   "L6",
    "src/l6_governance/risk_manager.py":                         "L6",
    "src/l6_governance/scaling.py":                              "L6",
    # ---- L7: l7_observability ----
    "src/l7_observability/alert_system.py":                      "L7",
    "src/l7_observability/apm_instrument.py":                    "L7",
    "src/l7_observability/dashboard.py":                         "L7",
    "src/l7_observability/health_check.py":                      "L7",
    "src/l7_observability/health_check_manager.py":              "L7",
    "src/l7_observability/health_endpoints.py":                  "L7",
    "src/l7_observability/health_monitor.py":                    "L7",
    "src/l7_observability/jaeger_tracer.py":                     "L7",
    "src/l7_observability/performance_monitor.py":               "L7",
    "src/l7_observability/prometheus_exporter.py":               "L7",
    # ---- L8: l8_lifecycle ----
    "src/l8_lifecycle/app_context.py":                           "L8",
    "src/l8_lifecycle/chaos_monkey.py":                          "L8",
    "src/l8_lifecycle/layer_orchestrator.py":                    "L8",
    "src/l8_lifecycle/lifecycle_manager.py":                     "L8",
    "src/l8_lifecycle/meta_controller.py":                       "L8",
    "src/l8_lifecycle/startup_orchestrator.py":                  "L8",
    "src/l8_lifecycle/watchdog.py":                              "L8",
}

# Directories to exclude from the scan.
EXCLUDE_DIRS = {
    ".venv", "venv", "__pycache__", ".git", ".archived", ".claude",
    ".mypy_cache", ".pytest_cache", "models", "snapshots", "logs",
    "artifacts", "validation_outputs", "data", "state",
    "tests",                # tests cross layers intentionally
}

# Module-name prefixes that come from the workspace.
WORKSPACE_TOP_LEVEL = {
    "core", "agents", "utils", "config", "automation", "monitoring",
    "dashboards", "diagnostics", "tools", "scripts", "stream",
    "portfolio",
}


def iter_python_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # in-place prune
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def module_to_path(modname: str) -> str | None:
    """Convert dotted module name to a relpath key in FILE_LAYER_MAP, if any."""
    parts = modname.split(".")
    if not parts or parts[0] not in WORKSPACE_TOP_LEVEL:
        return None
    candidate = "/".join(parts) + ".py"
    if candidate in FILE_LAYER_MAP:
        return candidate
    pkg_init = "/".join(parts) + "/__init__.py"
    if pkg_init in FILE_LAYER_MAP:
        return pkg_init
    return None


def parse_imports(path: Path) -> List[str]:
    """Return module names imported at *static* / module-load time.

    Excludes (treated as lazy / optional, not real layer dependencies):
      * Imports inside function or method bodies (late-bound).
      * Imports inside ``if TYPE_CHECKING:`` blocks (type-only).
      * Imports inside ``try: ... except ImportError:`` *fallback* blocks
        when the import is inside a function (already covered above).

    Top-level ``try/except ImportError`` import attempts ARE included
    because they execute at import time and create real coupling.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    out: List[str] = []

    def _is_type_checking_test(test: ast.AST) -> bool:
        # Matches `if TYPE_CHECKING:` and `if typing.TYPE_CHECKING:`
        if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
            return True
        if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
            return True
        return False

    def _walk(nodes, in_func: bool):
        for node in nodes:
            # Skip function/method bodies — imports there are late-bound.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # do NOT recurse into function body
            # Skip TYPE_CHECKING gated blocks entirely.
            if isinstance(node, ast.If) and _is_type_checking_test(node.test):
                continue
            # Class bodies count (decorators / base classes execute at import).
            if isinstance(node, ast.ClassDef):
                _walk(node.body, in_func)
                continue
            # If / Try / With at module scope: recurse into their bodies.
            if isinstance(node, ast.If):
                _walk(node.body, in_func)
                _walk(node.orelse, in_func)
                continue
            if isinstance(node, ast.Try):
                _walk(node.body, in_func)
                for h in node.handlers:
                    _walk(h.body, in_func)
                _walk(node.orelse, in_func)
                _walk(node.finalbody, in_func)
                continue
            if isinstance(node, (ast.With, ast.AsyncWith)):
                _walk(node.body, in_func)
                continue
            # Actual imports
            if isinstance(node, ast.Import):
                for n in node.names:
                    out.append(n.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    out.append(node.module)

    _walk(tree.body, in_func=False)
    return out


def check() -> Tuple[int, List[str], List[str]]:
    violations: List[str] = []
    unmapped:   List[str] = []

    for path in iter_python_files(ROOT):
        rel = str(path.relative_to(ROOT)).replace(os.sep, "/")
        caller_layer = FILE_LAYER_MAP.get(rel)
        if caller_layer is None:
            unmapped.append(rel)
            continue

        for mod in parse_imports(path):
            target = module_to_path(mod)
            if target is None:
                continue
            callee_layer = FILE_LAYER_MAP.get(target)
            if callee_layer is None:
                continue
            if callee_layer == caller_layer:
                continue
            allowed = ALLOWED_DEPENDENCIES.get(caller_layer, set())
            if callee_layer not in allowed:
                violations.append(
                    f"{rel} ({caller_layer})  →  {target} ({callee_layer})  "
                    f"FORBIDDEN  [allowed: {sorted(allowed) or 'none'}]"
                )

    return (1 if violations else 0), violations, unmapped


def main() -> int:
    rc, violations, unmapped = check()

    print(f"Mapped files:    {len(FILE_LAYER_MAP)}")
    print(f"Unmapped files:  {len(unmapped)}")
    if unmapped:
        print("  (unmapped — add to FILE_LAYER_MAP):")
        for u in sorted(unmapped):
            print(f"    · {u}")

    # Baseline support: pre-existing violations are tolerated until fixed,
    # but no new ones are allowed to slip in.
    baseline_path = ROOT / "scripts" / "layer_violations_baseline.txt"
    baseline: set[str] = set()
    if baseline_path.exists():
        baseline = {
            line.strip() for line in baseline_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

    new_violations  = [v for v in violations if v not in baseline]
    fixed_baseline  = sorted(baseline - set(violations))

    if "--write-baseline" in sys.argv:
        unique = sorted(set(violations))
        baseline_path.write_text(
            "# Auto-generated by scripts/check_layer_imports.py --write-baseline\n"
            "# Each line is a known layer-violation that the codebase already had\n"
            "# at the time the 8-layer architecture was introduced. New violations\n"
            "# are rejected; lines should be removed as the underlying imports get\n"
            "# refactored.\n"
            + "\n".join(unique) + "\n"
        )
        print(f"\n📝 Wrote baseline ({len(unique)} entries) → {baseline_path}")
        return 0

    if violations:
        print(f"\nTotal violations: {len(violations)}  "
              f"(baseline: {len(baseline)},  new: {len(new_violations)},  "
              f"fixed since baseline: {len(fixed_baseline)})")

    if fixed_baseline:
        print("\n🎉 Baseline entries that have been fixed (remove from baseline):")
        for v in fixed_baseline:
            print(f"  - {v}")

    if new_violations:
        print(f"\n❌ {len(new_violations)} NEW layer-violation(s) — reject:\n")
        for v in new_violations:
            print(f"  {v}")
        return 1

    if violations:
        print("\n⚠️  All violations are in baseline (tolerated, refactor pending).")
    else:
        print("\n✅ No layer violations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
