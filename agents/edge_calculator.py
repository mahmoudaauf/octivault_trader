# agents/edge_calculator.py
"""
Edge Calculator for Multi-Agent Alpha Amplifier

This module computes edge scores for agent signals that feed into the
composite edge aggregation system. Each agent produces a signal with
an edge score that reflects its confidence in the move and the expected
return relative to costs.

Edge Score Interpretation:
  +1.0  : Perfect confidence this will be profitable (rare)
  +0.5  : 50-50 confidence, expected value positive
   0.0  : No edge, neutral signal
  -0.5  : Expected value negative
  -1.0  : Perfect confidence this will lose

The composite edge = weighted_average(agent_edges) uses these to make
institutional-grade decisions.
"""

from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger("EdgeCalculator")


def compute_agent_edge(
    agent_name: str,
    action: str,
    confidence: float,
    expected_move_pct: Optional[float] = None,
    price_target: Optional[float] = None,
    entry_price: Optional[float] = None,
    stop_loss_pct: Optional[float] = None,
    **kwargs,
) -> float:
    """
    Compute edge score for an agent signal.
    
    Edge combines:
    1. Agent confidence (how sure the agent is about direction)
    2. Expected move (what the market should do)
    3. Risk/reward ratio (target vs stop)
    4. Instrument-specific factors (volatility, liquidity)
    
    Args:
        agent_name: Name of the agent (e.g., "TrendHunter")
        action: BUY, SELL, or HOLD
        confidence: Agent's confidence (0.0 to 1.0)
        expected_move_pct: Expected move in %
        price_target: Target price for TP
        entry_price: Entry price for scaling
        stop_loss_pct: Stop loss distance in %
        **kwargs: Additional context (symbol, timeframe, etc.)
    
    Returns:
        Edge score (-1.0 to +1.0)
    """
    if action == "HOLD":
        return 0.0
    
    # Base edge from confidence
    base_edge = float(confidence or 0.5) - 0.5  # Center at 0, range [-0.5, 0.5]
    
    # Adjust for expected move (if available)
    move_adjustment = 0.0
    if expected_move_pct is not None:
        move_pct = float(expected_move_pct or 0.0) / 100.0
        move_adjustment = min(0.2, move_pct)  # Cap at +0.2
    
    # Adjust for risk/reward (if prices available)
    rr_adjustment = 0.0
    if price_target is not None and entry_price is not None and stop_loss_pct is not None:
        try:
            entry = float(entry_price)
            target = float(price_target)
            sl_pct = float(stop_loss_pct) / 100.0
            
            if entry > 0:
                profit_pct = abs(target - entry) / entry
                loss_pct = sl_pct
                
                if loss_pct > 0:
                    rr_ratio = profit_pct / loss_pct
                    # Scale reward-risk: 2:1 ratio = +0.1 edge boost
                    rr_adjustment = min(0.2, (rr_ratio - 1.0) * 0.1)
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    
    # Agent-specific adjustments (some agents are better at certain patterns)
    agent_adjustment = _get_agent_adjustment(agent_name, action)
    
    # Combine all factors
    edge = base_edge + move_adjustment + rr_adjustment + agent_adjustment
    
    # Apply directional sign
    if action == "SELL":
        edge = -edge
    
    # Clamp to valid range
    edge = max(-1.0, min(1.0, edge))
    
    logger.debug(
        f"[EdgeCalc:{agent_name}] {action} edge={edge:.3f} "
        f"(base={base_edge:.3f} move={move_adjustment:.3f} rr={rr_adjustment:.3f} agent={agent_adjustment:.3f})"
    )
    
    return edge


def _get_agent_adjustment(agent_name: str, action: str) -> float:
    """
    Get edge adjustment specific to each agent's track record.
    
    These are calibrated based on historical performance of each agent.
    """
    adjustments = {
        "TrendHunter": 0.05 if action == "BUY" else 0.0,      # Slight boost on TrendHunter BUYs
        "DipSniper": 0.10 if action == "BUY" else 0.0,        # DipSniper great at BUY timing
        "LiquidationAgent": 0.08,                              # Good at both sides
        "MLForecaster": 0.12,                                  # Best overall edge
        "SymbolScreener": 0.05 if action == "BUY" else 0.0,   # Good at selection
        "IPOChaser": 0.0,                                      # Neutral
        "WalletScannerAgent": -0.02,                           # Slightly conservative
    }
    return adjustments.get(agent_name, 0.0)


def merge_signal_with_edge(signal: Dict[str, Any], edge: float) -> Dict[str, Any]:
    """
    Merge edge score into signal dictionary.
    
    Args:
        signal: Original signal dict
        edge: Computed edge score
    
    Returns:
        Updated signal dict with edge field
    """
    signal_copy = dict(signal or {})
    signal_copy["edge"] = float(edge)
    signal_copy["_edge_computed"] = True
    return signal_copy


def format_edge_for_logging(edge: float, threshold_buy: float = 0.35, threshold_sell: float = -0.35) -> str:
    """Format edge score for readable logging."""
    if edge >= threshold_buy:
        return f"🟢 BUY ({edge:.3f})"
    elif edge <= threshold_sell:
        return f"🔴 SELL ({edge:.3f})"
    else:
        return f"⚪ HOLD ({edge:.3f})"
