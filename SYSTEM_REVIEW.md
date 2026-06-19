# OctiVault Trader — Complete System Review
*Built from codebase: `core_engine/native/` + `agents/`*
*Date: 2026-06-19 | Branch: phase-3/wiring*

---

## 1. Signal Source (Single Active Path)

Only **one** signal source feeds the live pipeline:

| Component | File | Status |
|---|---|---|
| `MLForecaster` | `agents/ml_forecaster.py` | ✅ LIVE |
| `SymbolScreener` | `agents/symbol_screener.py` | ✅ LIVE (discovery only) |
| NativeSignalEngine (RSI/MACD/MA/Momentum) | `core_engine/native/signals.py` | ❌ NOT wired into main.py |
| All legacy agents (7 files) | `_archive/` | ❌ Archived |

### MLForecaster Details
- **Model type:** Keras/TensorFlow per-symbol `.keras` file (e.g. `models/mlforecaster_HBARUSDT_5m.keras`)
- **Features:** 31 inputs — OHLCV, `volume_zscore`, `volatility_zscore`, `taker_buy_vol_zscore`, `range_zscore`, plus lagged versions
- **Lookback:** 60 bars (5m candles = last 5 hours)
- **Persistence gate:** `ML_BUY_PERSIST_BARS=2` — signal must appear in **2 consecutive** 30s cycles before it fires a BUY
- **Output:** confidence score 0.0–1.0 per symbol

---

## 2. Gate Pipeline (BUY Signals — 11 Sequential Gates)

BUY signals pass through **all 11 gates**. First failure = blocked.

### Gate 1 — Symbol Format
- **Rule:** Symbol must end in `USDT` and be alphanumeric only
- **Purpose:** Sanity filter, prevents typos

---

### Gate 2 — Confidence Floor
- **Base floors by mode:**

| Mode | Confidence Floor |
|---|---|
| BOOTSTRAP (current — NAV < $100) | 0.50 |
| RECOVERY | 0.50 |
| NORMAL | 0.45 |
| GROWTH | 0.40 |
| SAFE (manual) | 0.90 |
| PROTECTIVE (manual) | 0.60 |
| PAUSED (manual) | 1.00 (blocks all) |

- **Regime adjustments on top of base floor:**

| Regime | Floor Delta |
|---|---|
| CHOPPY / RANGING | +0.10 |
| UNKNOWN | +0.05 |
| TRENDING | −0.05 |
| VOLATILE / CRISIS | +0.15 |
| UPTREND | −0.03 |

- **OFC delta:** OFC controller adjusts floor additively between **0.50 and 0.72** based on hourly P&L vs +2%/day target
- **Dynamic floor delta** (from arbitration engine): win rate <40% → +0.08, <50% → +0.04; DOWNTREND regime → +0.08; CHOPPY/VOLATILE → +0.04; recent SL on symbol → +0.05; 3+ global SLs in 1h → +0.08

---

### Gate 3 — Regime Gate (MODIFIED ✅)
**File:** `core_engine/native/regime_gate.py`

| Condition | Result |
|---|---|
| regime = `crisis` / `halted` / `low_liquidity` | ❌ Block (all signals) |
| regime = `range` (BUY only) | ❌ Block — no momentum for buy |
| Per-symbol regime = RANGING or CHOPPY (BUY only) | ❌ Block |
| Spread > 0.5% | ❌ Block (bad fill) |
| Liquidity score < 0.15 | ❌ Block |
| Volatile / vol_score ≥ 0.85 | ✅ Allow but +0.10 confidence floor bump |
| Order-book imbalance (bid% < 35%) | 🟡 Observe-only by default (`ORDERBOOK_IMBALANCE_VETO_ENABLED=false`) |
| Everything else | ✅ Allow |

**SELL signals skip regime check** — regime blocks BUY entries only.

---

### Gate 4 — Position Limit
- **Max concurrent positions:** 3 (BOOTSTRAP mode) — `max_concurrent_positions` env
- **Re-buy block threshold:** $2 notional (prevents double-buying same symbol after price decay)
- **Slot counting:** only positions ≥$10 notional count toward the limit
- **BNB dust exclusion:** BNB position < $1 doesn't consume a slot

---

### Gate 5 — Capital Availability
- **Rule:** Spendable USDT must be ≥ $10 (`min_notional_usdt`)
- **Spendable = free_USDT × (1 − 10% reserve ratio)**
- Returns False immediately if no free USDT exists

---

### Gate 6 — Risk Manager
Checks **in order:**
1. `trading_halted` flag (set by OFC kill-switch at 5% drawdown)
2. NAV protection mode: `FREEZE_BUY` or `RECOVERY` → block new BUYs
3. `max_drawdown_pct` exceeded (session drawdown check)
4. `daily_loss_limit_pct` exceeded
5. Total exposure ≥ 60% of NAV → block new BUYs

**SELL signals:** only checks `trading_halted` is False (exposure check skipped — high exposure is exactly when we need to sell).

---

### Gate 7 — Reentry Cooldown *(BUY only)*
Three-layer cooldown, all persisted to `logs/arb_state.json` (survives restarts):

| Trigger | Cooldown |
|---|---|
| SL exit on this symbol | 2 hours |
| Loss streak ≥ 3 consecutive losses | 4 hours |
| Normal post-buy | 15 minutes |

- Loss streak **resets** after 24h without a trade on that symbol
- After extended cooldown expires, streak is reset and symbol gets a fresh start

---

### Gate 8 — Symbol Performance Tracker *(BUY only)*
- Per-symbol win-rate tracker using closed trade history
- Blocks symbols with poor performance record
- Returns `size_multiplier` 0.0–1.25 for position sizing

---

### Gate 9 — Global Pace *(BUY only)*
Adaptive pace window — scales with overall win rate:

| Win Rate | Window | Max BUYs |
|---|---|---|
| ≥ 70% | 15 min | 4 |
| ≥ 60% | 15 min | 3 |
| ≥ 50% | 30 min | 3 |
| ≥ 40% | 45 min | 3 |
| < 40% | 60 min | 2 |

**Circuit breaker:** 2+ SL exits within any 1h window → block new BUYs for 1h after the most recent SL

---

### Gate 10 — No Average Down *(BUY only)*
- Checks if we already hold the symbol
- If current price < entry price AND global regime = DOWNTREND → **block**
- Closes the loophole where a declining position drops below $10 and would allow a second buy (concentration bug)
- Fail-open: if no position data or price unavailable → allows trade

---

### Gate 11 — Symbol Downtrend Veto *(BUY only)*
- Checks the **symbol's own price trend**, not the global regime
- Blocks if: price is **0.5% below its 20-bar MA** AND the MA is **falling** (vs 5 bars ago)
- Uses 5m timeframe, 20-bar MA window
- Closes the gap left by Gate 10 (which only blocks averaging-down in DOWNTREND)
- Fail-open: < 25 bars of data → allows trade

---

### SELL Signal Gates
Only **Gate 1** (format) and **Gate 6** (risk manager, exposure-skipped) apply.

---

## 3. TP/SL Engine
**File:** `core_engine/native/tp_sl_engine.py`

### Base Calculation (ATR-based)
```
SL_pct = 1.5 × ATR_14  →  clamped [1.0%, 2.5%]
TP_pct = max(SL_pct × 2.0, 1.5 × ATR_14)  →  floored at 1.5%, capped at 6.0%
```
- **Minimum R/R: 2:1** (`TP_RR_MULT = 2.0`)
- Fee floor: minimum net profit = 0.5% above 0.2% round-trip fee

### Time-Based Tightening
| Age | Action |
|---|---|
| 2 hours | TP tightened to +1.5%, SL moved to break-even |
| 4 hours | TP tightened to +0.8% |
| 3 hours (default) | Force exit (configurable via `TPSL_FORCE_EXIT_H`) |

### Regime-Aware Trailing Stop
Trailing stop activates only after profit exceeds threshold:

| Regime | Activation | Trail Distance |
|---|---|---|
| UPTREND | +2.0% profit | 1.0% below peak |
| TRENDING | +1.5% profit | 0.8% below peak |
| RANGING | +0.5% profit | 0.4% below peak |
| CHOPPY | +0.5% profit | 0.3% below peak |
| DOWNTREND | +0.5% profit | 0.3% below peak |

*In RANGING/CHOPPY: trailing arms early to capture stalled winners (TP rarely hits in chop; +0.3–0.5% net is higher expectancy).*

### Protective Tightening
Tightens TP after 30 min in `FREEZE_BUY`/`RECOVERY` NAV protection mode.

### Startup Grace Period
90 second suppression after restart — prevents stale first-tick prices from triggering phantom exits on restored positions.

---

## 4. Position Sizing (Capital Allocator + ACE)
**Files:** `core_engine/native/capital_allocator.py`, `core_engine/native/adaptive_capital_engine.py`

### Sizing Formula
```
kelly_allocation = free_USDT × 0.25 (Kelly) × confidence × (risk_per_symbol_pct / 100)
position_USD = min(kelly_allocation, NAV × 5%, mode_max_trade_usdt)
```

### Adaptive Capital Engine (ACE) Adjustments
| Condition | Risk Multiplier |
|---|---|
| Win rate > bonus threshold | × 1.08 |
| Win rate < penalty threshold | × max(0.25, 0.90) |

### OFC Size Multiplier
`SIZE_MULTIPLIER` from OFC: range **0.50–1.50**, updated every 15 min based on actual vs target hourly P&L

### Capital Reserve
- **Quote reserve ratio:** 10% of free USDT always kept back
- **Min notional:** $10 per trade
- **BOOTSTRAP mode cap:** $20 per trade

---

## 5. NAV Protection Engine
**File:** `core_engine/native/nav_protection.py`

Three-tier protection keyed off **session drawdown** (from session anchor NAV, not all-time peak):

| Drawdown | Mode | Action |
|---|---|---|
| ≥ 2% | DEFENSIVE | Size reduction + confidence floor raise |
| ≥ 5% | FREEZE_BUY | No new BUYs; existing positions allowed to exit |
| ≥ 8% | RECOVERY | No new BUYs; TP/SL tightened; auto-reset anchor when fully in cash |

- Protection floor locked at ≥ 95% of peak NAV (`minimum_protection_floor_ratio`)
- Auto-reset: when in RECOVERY and all positions are closed (exposure=0), anchor resets to current NAV and drawdown counter resets to 0%

---

## 6. Objective Feedback Controller (OFC)
**File:** `core_engine/native/objective_feedback_controller.py`

| Parameter | Value |
|---|---|
| Heartbeat | Every 15 min (`CHECKPOINT_HEARTBEAT_S=900`) |
| Daily target | +2%/day |
| Hourly target | +0.0833%/h |
| Confidence floor range | 0.50 – 0.72 |
| Size multiplier range | 0.50 – 1.50 |
| Kill-switch drawdown | 5% (`OBJ_MAX_DRAWDOWN_PCT`) |
| Net edge floor | 5 bps average |

**Kill-switch:** At 5% OFC drawdown, sets `trading_halted=True` (blocks Gate 6 for all BUYs)
**Control loops:** PI controller — adjusts confidence floor and size multiplier based on pace error (actual vs target hourly %) and drawdown error

---

## 7. Fear & Greed Index Integration
**File:** `core_engine/native/fear_greed.py`

| Condition | Action |
|---|---|
| F&G ≤ 15 (first fetch, no prior reading) | Auto-pause BUYs (`pause_buys.flag`) |
| F&G drops ≥5pts to ≤20 | Auto-pause BUYs |
| F&G rises ≥3pts + BTC 2 consecutive green 1h candles | Auto-resume at **HALF SIZE** (SIZE_MULTIPLIER=0.5) |
| First 5 profitable trades after resume | Half-size mode |
| After 5 profitable trades | Full size restored |

Scale: 0–24 Extreme Fear | 25–44 Fear | 45–55 Neutral | 56–74 Greed | 75–100 Extreme Greed
**Current value: 14 (Extreme Fear)** — BUYs paused

---

## 8. Concentration Guard
**File:** `core_engine/native/concentration_guard.py`

**Max cluster exposure: 40% of NAV per cluster**

| Cluster | Symbols |
|---|---|
| BTC | BTCUSDT |
| ETH | ETHUSDT |
| L1 | BNB, SOL, ADA, AVAX, ATOM, DOT, NEAR |
| MEME | DOGE, SHIB, PEPE, FLOKI, BONK |
| STABLE | USDC, BUSD, TUSD, FDUSD |
| OTHER | Everything else |

Checks: `(current_cluster_quote + proposed_quote) / NAV > 40%` → Block

---

## 9. Mode System (Auto vs Manual)
**File:** `core_engine/native/mode_manager.py`

**Auto mode** (from NAV):

| NAV | Mode | Max Positions | Conf Floor | Max Trade |
|---|---|---|---|---|
| < $100 | BOOTSTRAP | 3 | 0.50 | $20 |
| $100 – $500 | RECOVERY | 5 | 0.50 | $50 |
| $500 – $2,000 | NORMAL | 5 | 0.45 | $150 |
| > $2,000 | GROWTH | 5 | 0.40 | 15% of NAV |

**Manual modes** (set via API/CLI):

| Mode | Max Positions | Conf Floor | Max Trade |
|---|---|---|---|
| SAFE | 1 | 0.90 | $30 |
| PROTECTIVE | 2 | 0.60 | $50 |
| PAUSED | 0 | 1.00 | $0 |

**Current mode: BOOTSTRAP** (NAV ~$57)

---

## 10. Symbol Screener
**File:** `agents/symbol_screener.py`

- Runs every **3,600s** (1 hour)
- Scans top **30 symbols** by 24h USDT volume
- Filter: min ATR ≥ **0.30%** (removes flatlines)
- Outputs: active candidate list for MLForecaster training/inference

---

## 11. Live Pipeline End-to-End

```
SymbolScreener (hourly)
        ↓
MLForecaster (every ~30s cycle per symbol)
  → 31 features, 60-bar lookback
  → 2-cycle persistence gate
        ↓
SignalManagerBridge
        ↓
NativeArbitrationEngine.evaluate()
  Gate 1: symbol format
  Gate 2: confidence floor (mode + regime + OFC)
  Gate 3: regime gate (blocks RANGING/CHOPPY/CRISIS BUYs)  ← MODIFIED
  Gate 4: position limit (max 3 @ BOOTSTRAP)
  Gate 5: capital check (≥$10 spendable)
  Gate 6: risk manager (drawdown, exposure, halted, NAV protection)
  Gate 7: reentry cooldown (15min post-buy, 2h post-SL, 4h post-loss-streak)
  Gate 8: symbol performance tracker
  Gate 9: global pace (adaptive, circuit-breaker)
  Gate 10: no average-down in DOWNTREND
  Gate 11: symbol downtrend veto (20-MA check)
        ↓
NativeDecisionEngine.decide()
  → Kelly sizing (0.25 × confidence × 20% risk_pct)
  → Mode max trade cap ($20 BOOTSTRAP)
  → Concentration guard (40% cluster limit)
        ↓
NativeCapitalAllocator.allocate_for_buy()
  → ACE risk multiplier (win-rate scaled)
  → OFC SIZE_MULTIPLIER (0.50–1.50)
  → F&G half-size multiplier (if recently resumed)
        ↓
Executor → Binance Market Order
        ↓
NativeTPSLEngine.arm()
  → ATR-based TP/SL (2:1 R/R minimum)
  → Regime-aware trailing stop
  → Time-based tightening (2h/4h/3h force-exit)
```

---

## 12. Current Live State

| Parameter | Value |
|---|---|
| Mode | BOOTSTRAP |
| NAV | ~$57.85 |
| Open positions | 1 (HBAR) |
| F&G | 14 — Extreme Fear → BUYs paused |
| Market regime | RANGING → Gate 3 blocking |
| Active blocks | gate_3_regime, gate_4_position_limit |
| Win/Loss record | 3W / 6L, net PnL −$0.28 |
| HBAR position | entry $0.07993, TP $0.08153 (+2.0%), SL $0.07913 (−1.0%), 2:1 R/R |

---

## 13. Identified Gaps & Recommendations

| # | Gap | Impact | Fix |
|---|---|---|---|
| 1 | F&G auto-pause blocks ALL BUYs in Extreme Fear — but Extreme Fear is historically best entry time | Misses contrarian entries | Add `BUFFETT_FEAR_OVERRIDE` flag: if F&G <20 AND BTC confirmed reversal AND signal conf >0.75 → allow selective BUY |
| 2 | NativeSignalEngine (RSI/MACD/MA/Momentum) is built but never called in main.py | 4 extra confirmations wasted | Wire as a vote layer or at minimum use for regime classification |
| 3 | MLForecaster has no fallback if model file missing for a symbol | Silent failures | Add symbol skip with warning if `.keras` file not found |
| 4 | Gate 11 uses MA on 5m bars — in a ranging market the 20-MA can be above price even on mild intraday pullbacks | Over-blocking valid entries in BOOTSTRAP | Consider raising MA bars to 50 or using EMA |
| 5 | Win rate 33% (3W/6L) — avg loss −0.82% vs avg win +0.40% | Negative expectancy | Fix in progress via regime gate (gate_3) — needs more data to confirm |
| 6 | OFC kill-switch (5% drawdown) + NAV FREEZE_BUY (5%) both trigger at same threshold | Redundant, can cause double-halt | Stagger: NAV FREEZE_BUY at 4%, OFC kill-switch at 6% |
| 7 | Gate 9 SL circuit breaker resets after 1h — only 2 SLs required | Low bar for circuit breaker | Raise to 3 SLs or extend to 2h window |
| 8 | No re-training trigger when win rate drops below threshold | Model drift undetected | Auto-retrain signal when 30-trade win rate < 35% |
