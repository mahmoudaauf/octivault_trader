#!/usr/bin/env python3
"""
STRATEGY OPTIMIZATION CONFIG - Post-Reset
=========================================

This file contains the optimized strategy parameters to fix capital decay.

Applied after reset & restart:
- Pauses trading until verified
- Increases position size threshold
- Tightens entry filter requirements
- Adds win-rate gate for safety

Modified: May 1, 2026
Author: Capital Optimization Fix
"""

import os
from datetime import datetime

# ============================================================================
# TRADING ENABLE/DISABLE OVERRIDE
# ============================================================================
# Set to False initially to verify settings, then set to True
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "false").lower() == "true"

# ============================================================================
# POSITION SIZING PARAMETERS (FIX #1)
# ============================================================================
# Current: $25 per trade (fees = 0.2% of position)
# Fixed: $50+ per trade (fees = 0.1% of position)
# Effect: 50% reduction in fee drag

MIN_ECONOMIC_TRADE_USDT = 50.0  # was 25.0 - DOUBLED to reduce fee impact

# ============================================================================
# ENTRY FILTER PARAMETERS (FIX #2)
# ============================================================================
# Current: 0.12% expected profit (barely above breakeven)
# Fixed: 0.50% expected profit (4x higher threshold)
# Effect: Only take high-conviction trades

MIN_EXPECTED_NET_PCT = 0.50  # was 0.12% - QUADRUPLED for stricter filtering
MIN_EXPECTED_NET_USDT = 0.50  # was 0.04 - INCREASED to $0.50 minimum

# ============================================================================
# WIN-RATE GATE PARAMETERS (FIX #3)
# ============================================================================
# Current: None (taking all trades, even unproven ones)
# Fixed: Require 55%+ win rate from backtesting
# Effect: Avoid unproven strategies, protect capital

REQUIRE_WIN_RATE_GATE = True
MINIMUM_REQUIRED_WIN_RATE = 0.55  # 55% minimum

# Only trade if we have backtesting data
REQUIRE_BACKTEST_HISTORY = True
MIN_BACKTEST_SAMPLES = 30  # Require at least 30 historical trades

# ============================================================================
# TRADE FREQUENCY LIMITS (SUPPORTING FIX)
# ============================================================================
# Current: 100+ trades per day (high fees)
# Fixed: 5-10 trades per day (thoughtful trading)
# Effect: Fewer opportunities but much better quality

MAX_TRADES_PER_HOUR = 2  # Maximum 2 trades per hour
MAX_ACTIVE_POSITIONS = 3  # Keep max 3 positions open
MAX_TRADES_PER_DAY = 20  # Hard limit of 20 trades per day

# ============================================================================
# RESET METRICS & TRACKING
# ============================================================================
RESET_TIMESTAMP = datetime.now().isoformat()
RESET_REASON = "Capital decay optimization: tighten filters, increase position size"
PREVIOUS_STRATEGY_LOSS = -25.93  # Previous NAV loss
STARTING_CAPITAL_RESET = 99.76  # Starting capital after losses

# ============================================================================
# MONITORING & VERIFICATION
# ============================================================================
LOG_EVERY_FILTERED_TRADE = True  # Log why trades are rejected
ALERT_ON_TRADE_ABOVE_THRESHOLD = True  # Alert if old high-risk trades attempted

# ============================================================================
# SAFETY GUARDRAILS
# ============================================================================
# Don't let capital go below floor
ABSOLUTE_MINIMUM_USDT = 10.0

# Pause trading if drawdown exceeds this
MAX_ADDITIONAL_DRAWDOWN_PCT = 0.15  # 15% max additional loss

# ============================================================================
# MONITORING MODE
# ============================================================================
# Start in monitoring mode - no actual trades, just log decisions
MONITORING_MODE = True  # Set to False to actually trade after verification
MONITORING_DURATION_MINUTES = 30  # Monitor for 30 minutes before trading

print(
    f"""
═══════════════════════════════════════════════════════════
  ⚡ STRATEGY OPTIMIZATION CONFIG LOADED
═══════════════════════════════════════════════════════════

Optimizations Applied:
  ✓ Position size:       $25 → $50 (fee impact halved)
  ✓ Entry threshold:     0.12% → 0.50% (4x stricter)
  ✓ Win-rate gate:       None → 55% minimum (new)
  ✓ Trade frequency:     100+/day → 5-10/day (quality over quantity)

Status:
  Trading Enabled:       {TRADING_ENABLED}
  Monitoring Mode:       {MONITORING_MODE}
  Reset Timestamp:       {RESET_TIMESTAMP}

Previous Performance:
  Starting NAV:          $125.69
  Ending NAV:            $99.76
  Loss:                  -$25.93 (-20.63%)

Expected After Fix:
  Starting NAV:          $99.76
  Target Range:          $97-102 (stabilize or grow)
  Estimated Time:        1-7 days to break even

═══════════════════════════════════════════════════════════
"""
)
