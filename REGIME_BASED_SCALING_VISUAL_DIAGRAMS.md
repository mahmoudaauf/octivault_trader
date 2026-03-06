# Regime-Based Scaling Visual Diagrams & Flowcharts

## 1. Signal Flow Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          MARKET ANALYSIS                                   │
│                  (OHLCV Data, Technical Indicators)                         │
└────────────────────┬─────────────────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ TrendHunter Agent (Signal Generation)                                       │
│ ┌──────────────────────────────────────────────────────────────────────┐  │
│ │ 1. Analyze price (MACD/EMA or ML model)                            │  │
│ │    → Determine trend direction                                     │  │
│ │    → Calculate confidence                                          │  │
│ │    → Generate action (BUY/SELL/HOLD)                              │  │
│ └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│ ┌──────────────────────────────────────────────────────────────────────┐  │
│ │ 2. Get 1h Regime (from shared_state volatility data)              │  │
│ │    → Detect regime type: trending, sideways, bear, high_vol       │  │
│ └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│ ┌──────────────────────────────────────────────────────────────────────┐  │
│ │ 3. Get Regime Scaling Factors (NEW)                               │  │
│ │    ┌──────────────────────────────────────────┐                    │  │
│ │    │ if regime == "sideways":                 │                    │  │
│ │    │   return {                               │                    │  │
│ │    │     "position_size_mult": 0.50,          │                    │  │
│ │    │     "tp_target_mult": 0.60,              │                    │  │
│ │    │     "excursion_requirement_mult": 1.40,  │                    │  │
│ │    │     "trail_mult": 0.90,                  │                    │  │
│ │    │     "confidence_boost": -0.05            │                    │  │
│ │    │   }                                       │                    │  │
│ │    └──────────────────────────────────────────┘                    │  │
│ └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│ ┌──────────────────────────────────────────────────────────────────────┐  │
│ │ 4. Apply Confidence Adjustment (NEW)                              │  │
│ │    adjusted_confidence = confidence + regime_scaling["conf_boost"] │  │
│ │    (Example: 0.72 - 0.05 = 0.67)                                  │  │
│ └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│ ┌──────────────────────────────────────────────────────────────────────┐  │
│ │ 5. Filter on Adjusted Confidence (CHANGED)                        │  │
│ │    if adjusted_confidence < min_conf:                             │  │
│ │        return  # Gate by confidence, not by regime                 │  │
│ └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│ ┌──────────────────────────────────────────────────────────────────────┐  │
│ │ 6. Emit Signal with Regime Scaling (NEW)                          │  │
│ │    signal = {                                                      │  │
│ │      "symbol": "ETHUSDT",                                         │  │
│ │      "action": "BUY",                                             │  │
│ │      "confidence": 0.72,                                          │  │
│ │      "quote_hint": 100.0,                                         │  │
│ │      "_regime": "sideways",           ← NEW                       │  │
│ │      "_regime_scaling": {...}         ← NEW                       │  │
│ │    }                                                               │  │
│ └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────┬──────────────────────────────────────────────────┘
                          │
                          ▼ (Signal with scaling factors)
┌────────────────────────────────────────────────────────────────────────────┐
│ MetaController (Decision Making) [PHASE 2 ⏭️]                             │
│ ┌──────────────────────────────────────────────────────────────────────┐  │
│ │ 1. Receive signal with _regime_scaling                             │  │
│ │                                                                      │  │
│ │ 2. Apply Position Size Scaling (NEW)                              │  │
│ │    regime_scaling = signal["_regime_scaling"]                     │  │
│ │    mult = regime_scaling["position_size_mult"]                    │  │
│ │    signal["quote_hint"] = signal["quote_hint"] * mult             │  │
│ │    (Example: 100.0 * 0.50 = 50.0)                                │  │
│ │                                                                      │  │
│ │ 3. Apply other gating (capital, frequency, etc.)                 │  │
│ │                                                                      │  │
│ │ 4. Send to ExecutionManager                                       │  │
│ └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────┬──────────────────────────────────────────────────┘
                          │
                          ▼ (Signal with adjusted position size)
┌────────────────────────────────────────────────────────────────────────────┐
│ ExecutionManager (Order Placement)                                          │
│ ┌──────────────────────────────────────────────────────────────────────┐  │
│ │ 1. Create BUY order at signal["quote_hint"] (50.0, scaled)         │  │
│ │                                                                      │  │
│ │ 2. Place on exchange                                               │  │
│ │                                                                      │  │
│ │ 3. Store position with metadata                                    │  │
│ │    position["_regime_scaling"] = regime_scaling                    │  │
│ │    position["_regime"] = "sideways"                                │  │
│ │                                                                      │  │
│ │ 4. Position created (filled)                                       │  │
│ └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────┬──────────────────────────────────────────────────┘
                          │
                          ▼ (Open position with regime metadata)
┌────────────────────────────────────────────────────────────────────────────┐
│ TP/SL Engine (Risk Management) [PHASE 3 ⏭️]                               │
│ ┌──────────────────────────────────────────────────────────────────────┐  │
│ │ 1. Calculate base TP distance (1.5% from volatility)              │  │
│ │    (This already uses regime from shared_state)                    │  │
│ │                                                                      │  │
│ │ 2. Apply TP Target Scaling (NEW)                                  │  │
│ │    mult = position["_regime_scaling"]["tp_target_mult"]            │  │
│ │    tp_distance = base_tp * mult                                    │  │
│ │    (Example: 1.5% * 0.60 = 0.9%)                                  │  │
│ │                                                                      │  │
│ │ 3. Calculate TP price = entry + tp_distance                       │  │
│ │                                                                      │  │
│ │ 4. Check Excursion Gate (NEW)                                     │  │
│ │    mult = position["_regime_scaling"]["excursion_mult"]            │  │
│ │    threshold = base_threshold * mult                               │  │
│ │    is_valid = (current_price - entry) >= threshold                │  │
│ │    (Sideways: 1.4x harder to trigger, trending: 0.85x easier)    │  │
│ │                                                                      │  │
│ │ 5. Monitor position continuously                                   │  │
│ └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────┬──────────────────────────────────────────────────┘
                          │
                          ▼ (Position monitored with regime-adjusted TP/SL)
┌────────────────────────────────────────────────────────────────────────────┐
│ ExecutionManager (Trailing Management) [PHASE 4 ⏭️]                        │
│ ┌──────────────────────────────────────────────────────────────────────┐  │
│ │ 1. Get position's trailing configuration                           │  │
│ │                                                                      │  │
│ │ 2. Apply Trailing Multiplier (NEW)                                │  │
│ │    mult = position["_regime_scaling"]["trail_mult"]                │  │
│ │    trailing_stop = high - (atr * 1.5 * mult)                      │  │
│ │                                                                      │  │
│ │ 3. Check if stop triggered                                        │  │
│ │    if current_price <= trailing_stop:                             │  │
│ │        exit_position()                                             │  │
│ │                                                                      │  │
│ │ 4. Continue until exit or TP                                      │  │
│ │    (Sideways: tighter trailing, trending: looser trailing)        │  │
│ └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Regime Scaling Matrix

```
┌─────────────────┬──────────────┬──────────────┬──────────────┬──────────┐
│ REGIME          │ POS SIZE     │ TP TARGET    │ EXCURSION    │ TRAILING │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────┤
│ Trending ✓      │ 1.00x ✓      │ 1.00x ✓      │ 0.85x ✓      │ 1.30x ✓  │
│ (Uptrend)       │ (Full size)  │ (Full TP)    │ (Easier)     │ (Loose)  │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────┤
│ High Vol        │ 0.80x        │ 1.05x        │ 1.00x        │ 1.20x    │
│                 │ (80% size)   │ (Wider TP)   │ (Normal)     │ (Moderate)
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────┤
│ Sideways ✗      │ 0.50x ✗      │ 0.60x ✗      │ 1.40x ✗      │ 0.90x ✗  │
│ (Choppy/Range)  │ (50% size)   │ (60% TP)     │ (Harder)     │ (Tight)  │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────┤
│ Bear ✗          │ 0.60x        │ 0.80x        │ 1.20x ✗      │ 0.95x    │
│                 │ (60% size)   │ (80% TP)     │ (Harder)     │ (V.Tight)│
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────┤
│ Normal/Unknown  │ 1.00x        │ 1.00x        │ 1.00x        │ 1.00x    │
│ (Default)       │ (Baseline)   │ (Baseline)   │ (Baseline)   │ (Base)   │
└─────────────────┴──────────────┴──────────────┴──────────────┴──────────┘

Legend:
✓ = Favorable regime (go big)
✗ = Unfavorable regime (be careful)
All multipliers are factors to apply to base values
```

---

## 3. Example: Sideways Trade Execution

```
SIGNAL GENERATION
├─ Input: ETHUSDT, MACD BUY, confidence 0.72
├─ Regime detected: "sideways" (from 1h volatility)
├─ Regime scaling: {pos: 0.5x, tp: 0.6x, exc: 1.4x, trail: 0.9x, conf: -5%}
└─ Output: BUY signal with scaling metadata

                        ↓

METACONTROLLER
├─ Input: Signal with quote_hint=$100, pos_mult=0.5
├─ Calculation: 100 * 0.5 = 50
└─ Output: Order for $50 (instead of $100)

                        ↓

EXECUTIONMANAGER
├─ Input: Order for $50 at market
├─ Action: Place BUY order
└─ Output: Position filled at $50

                        ↓

TP/SL ENGINE - TP Calculation
├─ Input: Entry @ 1800, ATR=5, base TP=0.9%
├─ Calculation: 1800 + (1800 * 0.9% * 0.6) = 1800 + 9.7 = 1809.7
├─ (TP distance scaled from 16.2 to 9.7 because sideways)
└─ Output: TP set at 1809.7

                        ↓

TP/SL ENGINE - Excursion Gate
├─ Input: Base threshold=100 bps, excursion_mult=1.4
├─ Calculation: 100 * 1.4 = 140 bps required
├─ (Position must move 140 bps to confirm, not just 100)
└─ Output: Position marked valid only if moves 140+ bps

                        ↓

EXECUTIONMANAGER - Trailing
├─ Input: Base trailing=1.5x ATR, trail_mult=0.9
├─ Calculation: 1.5 * 0.9 = 1.35x ATR
├─ (Trailing at 1.35x ATR, tighter than normal)
└─ Output: Trailing stops update every bar

                        ↓

MONITORING LOOP (Every Bar)
├─ Check: Current price vs TP (1809.7)
├─ Check: Current price vs Trailing SL
├─ Check: Position still valid (excursion gate)
├─ At TP: Exit with profit (~0.9% move)
├─ At Trailing SL: Exit with loss (tight)
└─ If sideways breaks: Tight trailing catches exit quickly

                        ↓

RESULT
├─ Position size: 50% of normal (lower risk in choppy market)
├─ TP target: Tighter (0.9% instead of 1.5%)
├─ Excursion gate: Harder (140bp instead of 100bp)
├─ Trailing: Tight (1.35x ATR instead of 1.5x ATR)
└─ Outcome: Appropriate risk management for sideways regime
```

---

## 4. Binary Gating vs Regime Scaling

```
BINARY GATING (OLD) - ALL OR NOTHING
═════════════════════════════════════

Signal: ETHUSDT BUY, conf=0.85 (high quality)
Regime: sideways

Decision:
  if regime == "sideways":
      return  # Block entire signal

Result: ✗ Missed trade (100% blocked)
Opportunity lost: 0.8% profit potential


REGIME SCALING (NEW) - ADAPTIVE
═══════════════════════════════

Signal: ETHUSDT BUY, conf=0.85 (high quality)
Regime: sideways

Decision:
  scaling = get_regime_scaling("sideways")
  adjusted_conf = 0.85 - 0.05 = 0.80
  if adjusted_conf >= min_conf:
      pos_size = 100 * 0.5 = 50  # 50% size
      tp = 1.5% * 0.6 = 0.9%
      execute()

Result: ✓ Captured trade (with reduced risk)
Actual result: +0.8% profit on 50% position = +0.4% account impact

COMPARISON
──────────
Binary:  0% profit (missed) / 0% drawdown (blocked)
Scaling: +0.4% profit (captured) / 0.5% max loss (scaled)

Winner: Regime Scaling (captures alpha + manages risk)
```

---

## 5. Implementation Dependency Chain

```
                          ┏━━━━━━━━━━━━━━━━┓
                          ┃  Phase 5: Cfg ┃
                          ┃  (Optional)   ┃
                          ┗━━━━━━━━━━━━━━━┛
                                  ▲
                                  │
                          ┏━━━━━━━▼━━━━━━━┓
                          ┃ Phase 4: Exec ┃
                          ┃  Trailing     ┃
                          ┗━━━━━━━▲━━━━━━━┛
                                  │
                    ┏─────────────┴──────────────┐
                    │                           │
            ┏━━━━━━┴━━━━━━┓          ┏━━━━━━┴━━━━━━┓
            ┃ Phase 3a:  ┃          ┃ Phase 3b:  ┃
            ┃ TP/SL TP   ┃          ┃ TP/SL Exc. ┃
            ┗━━━━━━▲━━━━━┛          ┗━━━━━━▲━━━━━┛
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                          ┏━━━━━━━▼━━━━━━━┓
                          ┃ Phase 2: Meta ┃
                          ┃ Pos Size      ┃
                          ┗━━━━━━━▲━━━━━━━┛
                                  │
                          ┏━━━━━━━▼━━━━━━━┓
                          ┃ Phase 1: TH   ┃
                          ┃ ✅ COMPLETE   ┃
                          ┗━━━━━━━━━━━━━━━┛

Critical Path (must do in order):
1 → 2 → 3a,3b (parallel) → 4 → 5 (optional)

Can start Phase 3a & 3b in parallel once Phase 2 done
Phase 5 optional (system works without it)
```

---

## 6. Scaling Factor Application Across System

```
SIGNAL FROM TRENDHUNTER
  └─ _regime_scaling = {
      "position_size_mult": 0.5,
      "tp_target_mult": 0.6,
      "excursion_requirement_mult": 1.4,
      "trail_mult": 0.9,
      "confidence_boost": -0.05
    }

  ├─────────────────────────────────────────────────────────────────────
  │
  ├─→ METACONTROLLER [PHASE 2]
  │   └─ Uses: position_size_mult
  │      Multiplies quote_hint by 0.5
  │      Order size: 100 → 50 USDT
  │
  ├─────────────────────────────────────────────────────────────────────
  │
  ├─→ TP/SL ENGINE [PHASE 3a]
  │   └─ Uses: tp_target_mult
  │      Multiplies TP distance by 0.6
  │      TP range: 1.5% → 0.9%
  │
  ├─────────────────────────────────────────────────────────────────────
  │
  ├─→ TP/SL ENGINE [PHASE 3b]
  │   └─ Uses: excursion_requirement_mult
  │      Multiplies threshold by 1.4
  │      Excursion gate: 100bp → 140bp
  │
  ├─────────────────────────────────────────────────────────────────────
  │
  ├─→ EXECUTIONMANAGER [PHASE 4]
  │   └─ Uses: trail_mult
  │      Multiplies trailing distance by 0.9
  │      Trailing: 1.5xATR → 1.35xATR
  │
  └─────────────────────────────────────────────────────────────────────

ALL SCALING FACTORS WORK TOGETHER:
- Position size: 50% (less capital at risk)
- TP target: 60% of normal (tighter profit taking)
- Excursion gate: 140% (harder to confirm)
- Trailing: 90% of normal (tighter protection)

Result: Conservative, risk-managed position for sideways market
```

---

## 7. Phase Status Timeline

```
TODAY                           NEXT WEEK                    2 WEEKS
┣━━━━━━━━━━━━━━━━━━━━━┫         ┣━━━━━━━━━━━━━━━━━━━━┫     ┣━━━━━━━━┫

Phase 1 ✅              Phase 2 ⏳  Phase 3 ⏳  Phase 4 ⏳  Phase 5 ⏳  Testing
DONE                   (1-2h)     (2-3h)     (2-3h)     (1-2h)    (2-4d)
TrendHunter

  Signals carrying      Positions  TP/SL     Trailing  Config   Backtest
  regime scaling        sized by   scaled    scaled    external. Analysis
  factors              regime     by regime by regime


PARALLEL WORK (can do 3a & 3b together):
   │
   ├─ Phase 3a (TP scaling) ──┐
   │                          ├─ Both need Phase 2 done first
   ├─ Phase 3b (Excursion) ───┘


DEPENDENCIES:
Phase 1 → Phase 2 → Phase 3a & 3b → Phase 4 → Phase 5 → Testing
(done)   (next)    (parallel)      (then)    (optional) (validate)
```

---

## 8. Risk vs Reward by Regime

```
HIGH REWARD (Trending)
├─ Full position size (1.0x)
├─ Full TP targets (1.0x)
├─ Loose trailing (1.3x ATR)
└─ Result: Maximize gains when trend is clear
   ETHUSDT in uptrend: Full 100 USDT position, 1.5% TP target
   
   │
   ├─ TP hit: +1.5% on full position
   │
   └─ Risk: False signals in trending → mitigated by high confidence


MEDIUM REWARD (High Vol)
├─ 80% position size
├─ 105% TP targets (slightly wider)
├─ Moderate trailing (1.2x ATR)
└─ Result: Balance protection with opportunity
   BTCUSDT in high vol: 80 USDT position, 1.575% TP target


LOW REWARD (Sideways) ✗
├─ 50% position size (HALF)
├─ 60% TP targets (TIGHT)
├─ Tight trailing (0.9x ATR)
└─ Result: Minimize loss in choppy market
   BNBUSDT in sideways: 50 USDT position, 0.9% TP target
   
   │
   ├─ TP hit: +0.9% on 50% position = +0.45% account
   │
   ├─ SL hit: -0.5% on 50% position = -0.25% account
   │
   └─ Risk: Lower absolute loss due to smaller size


DEFENSIVE (Bear)
├─ 60% position size
├─ 80% TP targets
├─ Very tight trailing (0.95x ATR)
└─ Result: Protect in downmarket
   ETHUSDT in bear: 60 USDT position, 1.2% TP target

   │
   ├─ Tight trailing catches reversals quickly
   │
   └─ Risk: Limited upside but protected downside


SCALING WISDOM:
Trending = Go Big (1.0x)
High Vol = Be Moderate (0.8x)
Sideways = Be Small (0.5x)  ← Avoid big losses in chop
Bear = Be Defensive (0.6x)  ← Protect what you have
```

---

## 9. Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OCTIVAULT TRADING SYSTEM                         │
│                    (With Regime-Based Scaling)                          │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│   MARKET DATA        │
│ (OHLCV, Technical)   │
└─────────────┬────────┘
              │
              ▼
┌──────────────────────────────────────────────┐
│  VOLATILITY ANALYZER (1h, 5m)                │
│  ├─ Calculate ATR, realized vol              │
│  ├─ Detect regime: trending, sideways, bear  │
│  └─ Feed volatility_state to shared_state    │
└─────────────┬────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ AGENTS (Signal Generation)                   [PHASE 1 ✅]            │
│                                                                        │
│ TrendHunter:                                                          │
│  ├─ Get market data                                                  │
│  ├─ Analyze trend (MACD/EMA or ML)                                   │
│  ├─ Detect 1h regime from shared_state                               │
│  ├─ Get regime scaling factors                                       │
│  ├─ Emit signal WITH _regime_scaling metadata                        │
│  │  Example: {action: "BUY", _regime: "sideways",                   │
│  │           _regime_scaling: {pos: 0.5x, ...}}                     │
│  │                                                                     │
│  └─→ Signals collected in agent manager buffer                       │
│                                                                        │
│ Other Agents (RSI, Momentum, etc):                                   │
│  └─ Similar pattern with regime scaling                              │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│ METACONTROLLER (Decision Making)             [PHASE 2 ⏭️]            │
│                                                                        │
│  Process each signal:                                                 │
│  ├─ Read _regime_scaling from signal                                 │
│  ├─ Apply position_size_mult to quote_hint                           │
│  │  (100 USDT * 0.5 = 50 USDT for sideways)                         │
│  ├─ Apply other gating (capital limit, frequency, etc)               │
│  ├─ Decide: EXECUTE or BLOCK                                         │
│  └─→ Send decision to ExecutionManager                               │
│                                                                        │
│  Storage:                                                             │
│  └─ Save scaling metadata in decision record                         │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│ EXECUTIONMANAGER (Order Placement)                                    │
│                                                                        │
│  ├─ Receive decision with scaled position size                       │
│  ├─ Validate against min_notional, max_position, etc                 │
│  ├─ Place order on exchange at scaled size                           │
│  ├─ Store position with regime metadata                              │
│  │  (Store _regime_scaling in position record)                       │
│  │                                                                     │
│  └─→ Position now open and monitored                                 │
│                                                                        │
│  Continuous Monitoring:                                              │
│  ├─ Check TP/SL triggers                                             │
│  ├─ Check trailing stop updates                                      │
│  └─ Execute exits when conditions met                                │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│ TP/SL ENGINE (Risk Management)               [PHASE 3 ⏭️]            │
│                                                                        │
│  For each open position:                                              │
│  │                                                                     │
│  ├─ Read position metadata (_regime_scaling)                          │
│  │                                                                     │
│  ├─ Calculate TP Distance [PHASE 3a]                                 │
│  │  ├─ Base: volatility profile TP %                                 │
│  │  ├─ Apply tp_target_mult from _regime_scaling                     │
│  │  │  (1.5% * 0.6 = 0.9% for sideways)                             │
│  │  └─ Set TP price                                                  │
│  │                                                                     │
│  ├─ Check Excursion Gate [PHASE 3b]                                  │
│  │  ├─ Base: ATR * 0.35                                              │
│  │  ├─ Apply excursion_mult from _regime_scaling                     │
│  │  │  (100bp * 1.4 = 140bp for sideways)                            │
│  │  ├─ Check: current_excursion >= threshold?                        │
│  │  └─ Mark position as "valid" or "invalidated"                     │
│  │                                                                     │
│  ├─ Calculate SL                                                     │
│  │  ├─ Static SL from volatility profile                             │
│  │  └─ Trailing SL (updated by ExecutionManager)                     │
│  │                                                                     │
│  └─→ Continuous monitoring of TP/SL/gate conditions                  │
│                                                                        │
│  Exit Triggers:                                                       │
│  ├─ Price >= TP → Exit with profit                                   │
│  ├─ Price <= SL → Exit with loss                                     │
│  ├─ Trailing SL triggered → Exit with trail loss                     │
│  ├─ Excursion gate failed → Exit on false signal                     │
│  └─ Force close (risk) → Exit on emergency                           │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│ EXECUTIONMANAGER - Trailing Update [PHASE 4 ⏭️]                      │
│                                                                        │
│  Each bar for winning positions:                                      │
│  │                                                                     │
│  ├─ Get position's regime scaling (_regime_scaling)                   │
│  ├─ Calculate trailing distance                                      │
│  │  ├─ Base: atr * 1.5                                               │
│  │  ├─ Apply trail_mult from _regime_scaling                         │
│  │  │  (1.5 * 0.9 = 1.35x ATR for sideways)                         │
│  │  └─ Trailing SL = high - distance                                 │
│  │                                                                     │
│  ├─ Check if trailing triggered                                      │
│  │  └─ if price <= trailing_sl: EXIT                                 │
│  │                                                                     │
│  └─→ Position exits when trailing SL hit                             │
│                                                                        │
│  Outcome:                                                             │
│  ├─ Tight trailing in sideways (0.9x) = quick exit                   │
│  ├─ Loose trailing in trending (1.3x) = let it run                   │
│  └─ Different aggressiveness by regime                               │
└────────────────────────────────────────────────────────────────────┘

CONFIGURATION [PHASE 5 ⏭️]
├─ TREND_POSITION_SIZE_MULT_SIDEWAYS = 0.50
├─ TREND_TP_TARGET_MULT_SIDEWAYS = 0.60
├─ TREND_EXCURSION_MULT_SIDEWAYS = 1.40
├─ TREND_TRAIL_MULT_SIDEWAYS = 0.90
└─ ... (similar for other regimes)

MONITORING & METRICS
├─ Win rate by regime
├─ Profit factor by regime
├─ Sharpe ratio by regime
├─ Average position size by regime
└─ Scaling factor usage tracking
```

---

This visual documentation helps understand the complete flow of regime-based scaling through the entire system. Each phase adds a new scaling integration point that makes the system more regime-aware and responsive to market conditions.

