# 📊 Confidence Band Trading - Visual Diagrams

---

## 1. Confidence Band Spectrum

```
Confidence Spectrum (0.0 to 1.0)
│
│  ❌ REJECTED    ├─ MEDIUM BAND ─┤  STRONG BAND  ├─ VERY STRONG
│                 │                │                │
│ 0.0           0.56              0.70            1.0
│  └─────────────┴────────────────┴────────────────┴─
│                 
│  Below Medium   Medium (50%)     Strong (100%)
│  Confidence     Confidence       Confidence
│  TRADE NO       TRADE YES        TRADE YES
│  size: 0 USDT   size: 15 USDT    size: 30 USDT

Example: required_conf = 0.70
  strong_conf = 0.70
  medium_conf = 0.56 (= 0.70 × 0.8)
```

---

## 2. Signal Processing Pipeline

```
┌─────────────────────────────────────┐
│         Agent Generates Signal      │
│  (confidence, symbol, planned_quote)│
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│   _passes_tradeability_gate()       │
│                                     │
│  strong_conf = required_conf        │
│  medium_conf = required_conf × 0.8  │
│                                     │
│  Is conf >= strong_conf? ──YES─→ ✅
│  │                            scale=1.0
│  NO
│  │
│  Is conf >= medium_conf? ──YES─→ ✅
│  │                            scale=0.5 ← NEW!
│  NO
│  │
│  └──→ ❌ REJECT
│       scale=N/A
└──────────────┬──────────────────────┘
               │
               ↓ (if ✅)
┌─────────────────────────────────────┐
│   _execute_decision()               │
│                                     │
│  planned_quote = 30.0               │
│  position_scale = 1.0 or 0.5        │
│                                     │
│  if scale < 1.0:                    │
│    planned_quote *= scale           │
│    30.0 × 0.5 = 15.0                │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│   ExecutionManager.execute()        │
│                                     │
│  Execute at calculated quote:       │
│  - 30.0 USDT (strong band) or       │
│  - 15.0 USDT (medium band)          │
└─────────────────────────────────────┘
```

---

## 3. Trade Distribution Expected

### Before Confidence Bands
```
Total Signals: 100

confidence >= 0.70  (Strong)        40 signals   ─→ 40 trades ✓
0.56 < confidence < 0.70 (Medium)   30 signals   ─→ ❌ 0 trades
confidence <= 0.56  (Weak)          30 signals   ─→ ❌ 0 trades
                                                    ───────────
                                                    Total: 40 trades
```

### After Confidence Bands
```
Total Signals: 100

confidence >= 0.70  (Strong)        40 signals   ─→ 40 trades @ 100%  = 40 trades ✓
0.56 < confidence < 0.70 (Medium)   30 signals   ─→ 30 trades @ 50%   = 30 trades ✓ NEW!
confidence <= 0.56  (Weak)          30 signals   ─→ ❌ 0 trades       = 0 trades
                                                                         ───────────
                                                    Total: 70 trades (+75%!)
```

**Impact:** 75% more trading events (40 → 70 trades)

---

## 4. Position Size Allocation

### Strong Band Signals (confidence >= 0.70)
```
planned_quote: 30.00 USDT
position_scale: 1.0
actual_quote: 30.00 × 1.0 = 30.00 USDT

┌──────────────────────────────┐
│     30 USDT POSITION         │
│       (100% size)            │
└──────────────────────────────┘
```

### Medium Band Signals (0.56 <= confidence < 0.70)
```
planned_quote: 30.00 USDT
position_scale: 0.5
actual_quote: 30.00 × 0.5 = 15.00 USDT

┌────────────────┐
│  15 USDT POS.  │
│   (50% size)   │
└────────────────┘
```

### Example Portfolio Using Both
```
Initial NAV: $100 USDT

Trade 1 (Strong): $30 USDT  ├─→ Balance: $70 USDT
Trade 2 (Strong): $20 USDT  ├─→ Balance: $50 USDT
Trade 3 (Medium): $15 USDT  ├─→ Balance: $35 USDT
Trade 4 (Medium): $15 USDT  ├─→ Balance: $20 USDT
Trade 5 (Strong): $20 USDT  └─→ Balance: $0 USDT (fully deployed)

Total Deployed: $100 (5 trades across both bands)
Mix: 3 strong (60 USDT) + 2 medium (30 USDT) = 100 USDT
```

---

## 5. Confidence Band Decision Matrix

```
┌────────────────────┬──────────────┬───────────┬───────────────┐
│ Confidence Value   │ Band         │ Pass?     │ Position Size │
├────────────────────┼──────────────┼───────────┼───────────────┤
│ 0.95 (Very High)   │ STRONG       │ ✅ YES    │ 100% (30 USD) │
│ 0.80 (High)        │ STRONG       │ ✅ YES    │ 100% (30 USD) │
│ 0.75 (High)        │ STRONG       │ ✅ YES    │ 100% (30 USD) │
│ 0.70 (Required)    │ STRONG/EDGE  │ ✅ YES    │ 100% (30 USD) │
├────────────────────┼──────────────┼───────────┼───────────────┤
│ 0.65 (Medium)      │ MEDIUM ← NEW │ ✅ YES    │ 50% (15 USD)  │
│ 0.62 (Medium)      │ MEDIUM ← NEW │ ✅ YES    │ 50% (15 USD)  │
│ 0.58 (Medium)      │ MEDIUM ← NEW │ ✅ YES    │ 50% (15 USD)  │
│ 0.56 (Medium/Edge) │ MEDIUM       │ ✅ YES    │ 50% (15 USD)  │
├────────────────────┼──────────────┼───────────┼───────────────┤
│ 0.50 (Below Req)   │ WEAK         │ ❌ NO     │ — (Rejected)  │
│ 0.40 (Low)         │ WEAK         │ ❌ NO     │ — (Rejected)  │
│ 0.30 (Very Low)    │ WEAK         │ ❌ NO     │ — (Rejected)  │
└────────────────────┴──────────────┴───────────┴───────────────┘

Assumptions:
  required_conf = 0.70
  strong_conf = 0.70
  medium_conf = 0.56
  default_planned_quote = 30.0
  medium_scale = 0.5
```

---

## 6. Timing & Execution Flow

```
T+0ms:  Agent generates signal
        confidence: 0.62
        planned_quote: 30.0
        
T+0.5ms: MetaController._passes_tradeability_gate()
        - Calculate bands (strong=0.70, medium=0.56)
        - Check: 0.62 >= 0.70? NO
        - Check: 0.62 >= 0.56? YES
        - Set: signal["_position_scale"] = 0.5
        - Return: (True, 0.70, "conf_medium_band")
        
T+1ms: MetaController._execute_decision()
        - Retrieve position_scale = 0.5
        - Apply: 30.0 × 0.5 = 15.0
        - Update: signal["_planned_quote"] = 15.0
        
T+1.5ms: ExecutionManager.execute()
        - Check risk, balance, exchange limits
        - Execute 15.0 USDT buy order
        
T+2ms: Order submitted to exchange
```

**Total overhead: ~1ms per signal**

---

## 7. Configuration Sensitivity

### Impact of Band Ratio (0.56 → 0.60 → 0.70)

```
required_conf = 0.70

With CONFIDENCE_BAND_MEDIUM_RATIO = 0.80
  medium_conf = 0.70 × 0.80 = 0.56
  Medium band: 0.56 to 0.70 (14% range)
  → Accepts signals in 0.56-0.69 range

With CONFIDENCE_BAND_MEDIUM_RATIO = 0.75
  medium_conf = 0.70 × 0.75 = 0.525
  Medium band: 0.525 to 0.70 (17.5% range)
  → Accepts signals in 0.525-0.69 range (LOOSER, more trades)

With CONFIDENCE_BAND_MEDIUM_RATIO = 0.85
  medium_conf = 0.70 × 0.85 = 0.595
  Medium band: 0.595 to 0.70 (10.5% range)
  → Accepts signals in 0.595-0.69 range (TIGHTER, fewer trades)
```

### Impact of Position Scale (0.3 → 0.5 → 0.7)

```
planned_quote = 30.0 USDT

With CONFIDENCE_BAND_MEDIUM_SCALE = 0.3
  Medium position = 30.0 × 0.3 = 9.0 USDT
  → Conservative (more capital available)
  
With CONFIDENCE_BAND_MEDIUM_SCALE = 0.5 (DEFAULT)
  Medium position = 30.0 × 0.5 = 15.0 USDT
  → Balanced (good capital utilization)
  
With CONFIDENCE_BAND_MEDIUM_SCALE = 0.7
  Medium position = 30.0 × 0.7 = 21.0 USDT
  → Aggressive (high capital deployment)
```

---

## 8. Profit & Loss Scenarios

### Scenario: 3 Medium-Band Trades

```
Initial Capital: $100 USDT

Trade 1: Medium band @ 0.62 confidence
  Entry:  15.0 USDT
  Exit:   +10% profit
  Profit: 1.50 USDT
  Balance: 101.50 USDT

Trade 2: Medium band @ 0.60 confidence
  Entry:  15.0 USDT
  Exit:   -5% loss
  Loss:   -0.75 USDT
  Balance: 100.75 USDT

Trade 3: Medium band @ 0.65 confidence
  Entry:  15.0 USDT
  Exit:   +8% profit
  Profit: 1.20 USDT
  Balance: 101.95 USDT

Summary:
  - 3 trades, avg confidence: 0.62
  - Total cost: 45.0 USDT
  - Win rate: 66% (2/3)
  - Total profit: 1.95 USDT (+1.95%)
  - Final balance: 101.95 USDT

Key: Even with <70% confidence, medium-band trades can be profitable!
```

---

## 9. Logging Output Examples

### Trade Accepted (Strong Band)
```
[Meta:ConfidenceBand] BTCUSDT strong band: conf=0.750 >= strong=0.700 (scale=1.0)
→ No scaling applied (1.0 = identity)
```

### Trade Accepted (Medium Band)
```
[Meta:ConfidenceBand] ETHUSDT medium band: 0.560 <= conf=0.620 < strong=0.700 (scale=0.50)
[Meta:ConfidenceBand] Applied position scaling to ETHUSDT: 30.00 → 15.00 (scale=0.50)
→ Position halved due to lower confidence
```

### Trade Rejected
```
[Meta:Tradeability] Skip ADAUSDT BUY: conf 0.45 < floor 0.70 (reason=conf_below_floor)
→ Below medium band, no trade executed
```

---

## 10. Performance Impact Graph

```
CPU Usage per Signal
│
│ 1ms ├─ _passes_tradeability_gate()     [0.5ms]
│     ├─ _execute_decision() scaling     [0.2ms]
│     ├─ ExecutionManager overhead       [0.3ms]
│     │
│ 0ms └─────────────────────────────────────
│     OLD SYSTEM (no scaling)  vs  NEW SYSTEM (with scaling)
│     Difference: <1ms (negligible)
│
│ Memory per signal:
│ OLD: signal dict (~500 bytes)
│ NEW: signal dict + _position_scale (~600 bytes)
│ Difference: <100 bytes
```

---

## 11. State Machine

```
Signal Received
    │
    ├─→ Pass _passes_tradeability_gate()?
    │   ├─ YES → Set _position_scale
    │   └─ NO → Reject & Return
    │
    ├─→ Pass position limit check?
    │   ├─ YES → Continue
    │   └─ NO → Reject & Return
    │
    ├─→ Pass risk pre-check?
    │   ├─ YES → Continue
    │   └─ NO → Reject & Return
    │
    ├─→ Apply position scaling
    │   ├─ If scale < 1.0: multiply planned_quote
    │   └─ Update signal["_planned_quote"]
    │
    ├─→ Pass ExecutionManager execution?
    │   ├─ YES → Order placed ✓
    │   └─ NO → Execution failed ✗
    │
    └─→ Log result
```

---

## 12. Capital Deployment Curves

### Before (No Medium Band)
```
Capital Used (%)
100% │                    ╱
     │                ╱
 80% │            ╱
     │        ╱
 60% │    ╱
     │╱
 40% │
     │
 20% │
     │
  0% └────────────────────────────
     0   10   20   30   40   50
        Signals Received (sorted by confidence)
     
Only signals above 0.70 confidence get trades
→ Steep curve, fast capital deployment until cutoff
```

### After (With Medium Band)
```
Capital Used (%)
100% │                        ╱
     │                    ╱
 80% │                ╱
     │            ╱
 60% │        ╱
     │    ╱
 40% │╱    (medium band trades here)
     │
 20% │
     │
  0% └────────────────────────────
     0   10   20   30   40   50
        Signals Received (sorted by confidence)
     
Signals from 0.56-0.70 also get trades (50% sized)
→ Smoother curve, consistent capital deployment
```

---

**These diagrams illustrate the complete confidence band trading system.**

Each diagram shows a different aspect:
1. Spectrum: How confidence maps to action
2. Pipeline: Complete signal processing flow
3. Distribution: Expected changes in trade volume
4. Allocation: Position sizes by band
5. Matrix: Quick reference for confidence levels
6. Timing: Execution speed (< 1ms overhead)
7. Sensitivity: How config changes affect behavior
8. P&L: Example profitability scenarios
9. Logs: What you'll see in system logs
10. Performance: CPU/memory impact
11. State Machine: Full execution flow
12. Deployment: Capital utilization over time
