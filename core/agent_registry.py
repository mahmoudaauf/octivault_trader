# agent_registry.py
import inspect
from agents.ipo_chaser import IPOChaser
from agents.dip_sniper import DipSniper
# from agents.swing_trade_hunter import SwingTradeHunter
from agents.trend_hunter import TrendHunter
# from agents.news_reactor import NewsReactor
# from agents.signal_fusion_agent import SignalFusion
from agents.liquidation_agent import LiquidationAgent
# from agents.arbitrage_hunter_agent import ArbitrageHunter
# MLForecaster is optional in some deployments; guard import to avoid hard failures
try:
    from agents.ml_forecaster import MLForecaster
except Exception:
    MLForecaster = None
from agents.wallet_scanner_agent import WalletScannerAgent # Added import
from agents.symbol_screener import SymbolScreener # Added import

AGENT_CLASS_MAP = {
    "IPOChaser": IPOChaser,
    "DipSniper": DipSniper,
    # "SwingTradeHunter": SwingTradeHunter,
    "TrendHunter": TrendHunter,
    # "NewsReactor": NewsReactor,
    # "SignalFusion": SignalFusion,
    "LiquidationAgent": LiquidationAgent,
    # "ArbitrageHunter": ArbitrageHunter,
    # Only include if available in this deployment
    **({"MLForecaster": MLForecaster} if MLForecaster else {}),
    "WalletScannerAgent": WalletScannerAgent, # Added to map
    "WalletScanner": WalletScannerAgent,      # Added alias
    "SymbolScreener": SymbolScreener,         # Added to map
}

def validate_agent_registry(required_args):
    """
    Validates that all agents in AGENT_CLASS_MAP accept the required constructor arguments,
    with support for **kwargs fallback.

    Args:
        required_args (list): A list of strings representing the required argument names
                              (e.g., ['shared_state', 'config']).

    Returns:
        list: A list of strings, where each string describes an issue found (e.g.,
              "AgentName missing required arg: arg_name"). Returns an empty list
              if no issues are found.
    """
    issues = []
    for name, agent_class in AGENT_CLASS_MAP.items():
        # Get the signature of the agent's __init__ method
        sig = inspect.signature(agent_class.__init__)
        # Check if the __init__ method accepts **kwargs
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        # Get the names of the parameters, skipping 'self'
        init_params = list(sig.parameters.keys())[1:]

        for arg in required_args:
            # If an argument is not explicitly in init_params AND the method does not accept **kwargs,
            # then it's a missing argument.
            if arg not in init_params and not accepts_kwargs:
                issues.append(f"{name} missing required arg: {arg}")
    return issues

if __name__ == "__main__":
    # Optional: Validate that at least 'shared_state' and 'config' are present
    required_args = ["shared_state", "config"]
    # Run the validation
    issues = validate_agent_registry(required_args)
    # Print the results of the validation
    if issues:
        for issue in issues:
            print("❌", issue)
    else:
        print("✅ All agent constructors are valid.")

# Proposer/Discovery Agent Registration (Phase 3)
def register_all_discovery_agents(agent_manager, app_context):
    """
    Registers discovery/proposer agents. If an expected agent instance
    wasn't pre-built on `app_context`, attempt to construct it here.
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
            return cls(**{k: v for k, v in kwargs.items() if v is not None})
        except TypeError:
            try:
                return cls(kwargs["shared_state"], kwargs["config"])
            except Exception:
                return None
        except Exception:
            return None

    wallet_scanner = getattr(app_context, "wallet_scanner_agent", None) or _safe_build("WalletScannerAgent")
    symbol_screener = getattr(app_context, "symbol_screener_agent", None) or _safe_build("SymbolScreener")
    ipo_chaser = getattr(app_context, "ipo_chaser", None) or _safe_build("IPOChaser")

    for agent in [wallet_scanner, symbol_screener, ipo_chaser]:
        if not agent:
            continue
        if not getattr(agent, "agent_type", None):
            setattr(agent, "agent_type", "discovery")  # tag it
        agent_manager.register_agent(agent)            # correct signature
        agent_manager.register_discovery_agent(agent)  # add to discovery list
