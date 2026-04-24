# agent_registry.py
import logging
import inspect
import traceback

__all__ = [
    "AGENT_CLASS_MAP",
    "AGENT_IMPORT_ERRORS",
    "validate_agent_registry",
    "register_all_discovery_agents",
    "register_all_strategy_agents",
]

_logger = logging.getLogger("AgentRegistry")
AGENT_IMPORT_ERRORS: dict = {}

# ---------------------------------------------------------------------------
# Fault-isolated agent imports
# Each agent is wrapped individually so one broken module cannot crash the
# entire registry.  A placeholder stub is kept in AGENT_CLASS_MAP so
# AgentManager sees the entry and emits an explicit registration failure
# rather than silently missing the agent.
# ---------------------------------------------------------------------------

try:
    from agents.ipo_chaser import IPOChaser
except Exception as _e:
    class IPOChaser:  # type: ignore
        """Placeholder — IPOChaser module failed to import."""
        agent_type = "discovery"

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "IPOChaser import failed: %s"
                % AGENT_IMPORT_ERRORS.get("IPOChaser", {}).get("error", "unknown")
            )

    AGENT_IMPORT_ERRORS["IPOChaser"] = {
        "error": repr(_e),
        "traceback": traceback.format_exc(),
    }
    _logger.warning("IPOChaser import failed; placeholder kept: %s", _e, exc_info=True)

try:
    from agents.dip_sniper import DipSniper
except Exception as _e:
    class DipSniper:  # type: ignore
        """Placeholder — DipSniper module failed to import."""
        agent_type = "strategy"

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DipSniper import failed: %s"
                % AGENT_IMPORT_ERRORS.get("DipSniper", {}).get("error", "unknown")
            )

    AGENT_IMPORT_ERRORS["DipSniper"] = {
        "error": repr(_e),
        "traceback": traceback.format_exc(),
    }
    _logger.warning("DipSniper import failed; placeholder kept: %s", _e, exc_info=True)

try:
    from agents.trend_hunter import TrendHunter
except Exception as _e:
    class TrendHunter:  # type: ignore
        """Placeholder — TrendHunter module failed to import."""
        agent_type = "strategy"

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TrendHunter import failed: %s"
                % AGENT_IMPORT_ERRORS.get("TrendHunter", {}).get("error", "unknown")
            )

    AGENT_IMPORT_ERRORS["TrendHunter"] = {
        "error": repr(_e),
        "traceback": traceback.format_exc(),
    }
    _logger.warning("TrendHunter import failed; placeholder kept: %s", _e, exc_info=True)

try:
    from agents.liquidation_agent import LiquidationAgent
except Exception as _e:
    class LiquidationAgent:  # type: ignore
        """Placeholder — LiquidationAgent module failed to import."""
        agent_type = "strategy"

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LiquidationAgent import failed: %s"
                % AGENT_IMPORT_ERRORS.get("LiquidationAgent", {}).get("error", "unknown")
            )

    AGENT_IMPORT_ERRORS["LiquidationAgent"] = {
        "error": repr(_e),
        "traceback": traceback.format_exc(),
    }
    _logger.warning("LiquidationAgent import failed; placeholder kept: %s", _e, exc_info=True)

try:
    from agents.wallet_scanner_agent import WalletScannerAgent
except Exception as _e:
    class WalletScannerAgent:  # type: ignore
        """Placeholder — WalletScannerAgent module failed to import."""
        agent_type = "discovery"

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "WalletScannerAgent import failed: %s"
                % AGENT_IMPORT_ERRORS.get("WalletScannerAgent", {}).get("error", "unknown")
            )

    AGENT_IMPORT_ERRORS["WalletScannerAgent"] = {
        "error": repr(_e),
        "traceback": traceback.format_exc(),
    }
    _logger.warning("WalletScannerAgent import failed; placeholder kept: %s", _e, exc_info=True)

try:
    from agents.symbol_screener import SymbolScreener
except Exception as _e:
    class SymbolScreener:  # type: ignore
        """Placeholder — SymbolScreener module failed to import."""
        agent_type = "discovery"

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SymbolScreener import failed: %s"
                % AGENT_IMPORT_ERRORS.get("SymbolScreener", {}).get("error", "unknown")
            )

    AGENT_IMPORT_ERRORS["SymbolScreener"] = {
        "error": repr(_e),
        "traceback": traceback.format_exc(),
    }
    _logger.warning("SymbolScreener import failed; placeholder kept: %s", _e, exc_info=True)

# MLForecaster is optional in some deployments; guard import to avoid hard failures
try:
    from agents.ml_forecaster import MLForecaster
except Exception as _e:
    class MLForecaster:  # type: ignore
        """
        Placeholder that preserves registry visibility when MLForecaster import fails.
        AgentManager will attempt registration and emit an explicit failure.
        """
        agent_type = "strategy"

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MLForecaster import failed: %s"
                % AGENT_IMPORT_ERRORS.get("MLForecaster", {}).get("error", "unknown")
            )

    AGENT_IMPORT_ERRORS["MLForecaster"] = {
        "error": repr(_e),
        "traceback": traceback.format_exc(),
    }
    _logger.warning(
        "MLForecaster import failed; placeholder kept in AGENT_CLASS_MAP for explicit registration failure: %s",
        _e,
        exc_info=True,
    )

# ---------------------------------------------------------------------------
# Agents currently disabled (commented out to suppress import overhead).
# Re-enable by uncommenting both the import and the AGENT_CLASS_MAP entry.
# ---------------------------------------------------------------------------
try:
    from agents.swing_trade_hunter import SwingTradeHunter
except Exception as _e:
    class SwingTradeHunter:  # type: ignore
        """Placeholder — SwingTradeHunter module failed to import."""
        agent_type = "strategy"

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SwingTradeHunter import failed: %s"
                % AGENT_IMPORT_ERRORS.get("SwingTradeHunter", {}).get("error", "unknown")
            )

    AGENT_IMPORT_ERRORS["SwingTradeHunter"] = {
        "error": repr(_e),
        "traceback": traceback.format_exc(),
    }
    _logger.warning("SwingTradeHunter import failed; placeholder kept: %s", _e, exc_info=True)

# from agents.news_reactor import NewsReactor              # disabled: news feed not wired
# from agents.signal_fusion_agent import SignalFusion      # disabled: replaced by SignalManager
# from agents.arbitrage_hunter_agent import ArbitrageHunter  # disabled: latency constraints

AGENT_CLASS_MAP: dict = {
    "IPOChaser": IPOChaser,
    "DipSniper": DipSniper,
    "SwingTradeHunter": SwingTradeHunter,
    "TrendHunter": TrendHunter,
    # "NewsReactor": NewsReactor,
    # "SignalFusion": SignalFusion,
    "LiquidationAgent": LiquidationAgent,
    # "ArbitrageHunter": ArbitrageHunter,
    "MLForecaster": MLForecaster,
    "WalletScannerAgent": WalletScannerAgent,
    "WalletScanner": WalletScannerAgent,  # alias — same class as WalletScannerAgent
    "SymbolScreener": SymbolScreener,
}


def validate_agent_registry(required_args):
    """
    Validates that all agents in AGENT_CLASS_MAP accept the required
    constructor arguments, with support for **kwargs fallback.

    Alias entries (multiple map keys pointing to the same class) are
    deduplicated — each class is inspected only once.

    Args:
        required_args (list): Required argument names (e.g. ['shared_state', 'config']).

    Returns:
        list: Descriptions of issues found; empty list if none.
    """
    issues = []
    seen_classes: set = set()
    for name, agent_class in AGENT_CLASS_MAP.items():
        # Skip alias duplicates — same class already validated under another key
        class_id = id(agent_class)
        if class_id in seen_classes:
            continue
        seen_classes.add(class_id)

        try:
            sig = inspect.signature(agent_class.__init__)
        except (ValueError, TypeError) as exc:
            issues.append("%s __init__ is not inspectable: %s" % (name, exc))
            continue

        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        init_params = list(sig.parameters.keys())[1:]  # drop 'self'

        for arg in required_args:
            if arg not in init_params and not accepts_kwargs:
                issues.append("%s missing required arg: %s" % (name, arg))

    return issues


if __name__ == "__main__":
    _required_args = ["shared_state", "config"]
    _issues = validate_agent_registry(_required_args)
    if _issues:
        for _issue in _issues:
            print("[FAIL]", _issue)
    else:
        print("[OK] All agent constructors are valid.")


# ---------------------------------------------------------------------------
# Proposer / Discovery Agent Registration
# ---------------------------------------------------------------------------

# Explicit list of discovery agents.  Add new agents here — do NOT rely on
# iterating AGENT_CLASS_MAP, which also contains non-discovery agents.
_DISCOVERY_AGENTS = [
    ("wallet_scanner_agent", "WalletScannerAgent"),
    ("symbol_screener", "SymbolScreener"),
    ("symbol_screener_agent", "SymbolScreener"),
    ("ipo_chaser", "IPOChaser"),
]


def register_all_discovery_agents(agent_manager, app_context):
    """
    Registers discovery/proposer agents with the agent manager.

    For each entry in ``_DISCOVERY_AGENTS`` the function first checks
    ``app_context`` for a pre-built instance, then falls back to constructing
    one via ``_safe_build``.  Only ``register_discovery_agent`` is called
    (not the general ``register_agent``) to avoid double-execution in the
    agent manager's strategy loop.
    """

    def _safe_build(name: str):
        cls = AGENT_CLASS_MAP.get(name)
        if not cls:
            return None
        kwargs = {
            "shared_state": getattr(app_context, "shared_state", None),
            "config": getattr(app_context, "config", None),
            "exchange_client": getattr(app_context, "exchange_client", None),
            "symbol_manager": getattr(app_context, "symbol_manager", None),
            "execution_manager": getattr(app_context, "execution_manager", None),
            "tp_sl_engine": getattr(app_context, "tp_sl_engine", None),
        }
        try:
            filtered = {k: v for k, v in kwargs.items() if v is not None}
            try:
                sig = inspect.signature(cls.__init__)
                params = set(sig.parameters.keys()) - {"self"}
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                if not accepts_var_kw:
                    filtered = {k: v for k, v in filtered.items() if k in params}
            except Exception:
                # Signature introspection failed; keep best-effort kwargs.
                pass
            return cls(**filtered)
        except Exception as exc:
            _logger.debug("[_safe_build] %s construction failed: %s", name, exc)
            return None

    for attr_name, map_key in _DISCOVERY_AGENTS:
        agent = getattr(app_context, attr_name, None) or _safe_build(map_key)
        if not agent:
            continue
        if not getattr(agent, "agent_type", None):
            setattr(agent, "agent_type", "discovery")
        agent_manager.register_discovery_agent(agent)


# ---------------------------------------------------------------------------
# Strategy Agent Registration
# ---------------------------------------------------------------------------

# Explicit list of strategy agents.  Add new agents here — do NOT rely on
# iterating AGENT_CLASS_MAP, which also contains non-strategy agents.
_STRATEGY_AGENTS = [
    ("ml_forecaster", "MLForecaster"),
    ("dip_sniper", "DipSniper"),
    ("trend_hunter", "TrendHunter"),
    ("swing_trade_hunter", "SwingTradeHunter"),
    ("news_reactor", "NewsReactor"),
]


def register_all_strategy_agents(agent_manager, app_context):
    """
    Registers strategy agents with the agent manager.

    For each entry in ``_STRATEGY_AGENTS`` the function first checks
    ``app_context`` for a pre-built instance, then falls back to constructing
    one via ``_safe_build``.  Only ``register_agent`` is called to allow
    strategy agents to run in the agent manager's strategy execution loop.
    """

    def _safe_build(name: str):
        cls = AGENT_CLASS_MAP.get(name)
        if not cls:
            return None
        kwargs = {
            "shared_state": getattr(app_context, "shared_state", None),
            "config": getattr(app_context, "config", None),
            "market_data": getattr(app_context, "market_data_feed", None),
            "exchange_client": getattr(app_context, "exchange_client", None),
            "symbol_manager": getattr(app_context, "symbol_manager", None),
            "execution_manager": getattr(app_context, "execution_manager", None),
            "tp_sl_engine": getattr(app_context, "tp_sl_engine", None),
            "model_manager": getattr(app_context, "model_manager", None),
            "meta_controller": getattr(app_context, "meta_controller", None),
            "market_data_feed": getattr(app_context, "market_data_feed", None),
        }
        try:
            filtered = {k: v for k, v in kwargs.items() if v is not None}
            try:
                sig = inspect.signature(cls.__init__)
                params = set(sig.parameters.keys()) - {"self"}
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                if not accepts_var_kw:
                    filtered = {k: v for k, v in filtered.items() if k in params}
            except Exception:
                pass
            return cls(**filtered)
        except Exception as exc:
            _logger.debug("[_safe_build] %s construction failed: %s", name, exc)
            return None

    for attr_name, map_key in _STRATEGY_AGENTS:
        agent = getattr(app_context, attr_name, None) or _safe_build(map_key)
        if not agent:
            continue
        if not getattr(agent, "agent_type", None):
            setattr(agent, "agent_type", "strategy")
        agent_manager.register_agent(agent)
