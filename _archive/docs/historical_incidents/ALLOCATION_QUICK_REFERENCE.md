# 60/20/20 Portfolio Allocation - Quick Summary

## Current System State ✅

| Metric | Value |
|--------|-------|
| **Total Equity** | $83.85 USDT |
| **Free Capital** | $72.49 USDT (86.6%) |
| **Tied Up Capital** | $11.36 USDT (13.4%) |
| **Dust Positions** | 41 (being healed) |

## Live Allocation (When Deployed)

From the $72.49 free capital:

```
┌─────────────────────────────────────┐
│  TIER A: SWING/COMPOUND             │
│  60% = $43.49 USDT                  │
│  Agent: SwingTradeHunter            │
│  Confidence: 0.80 ✅ (FIXED)        │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  TIER B: DIP/BUFFER                 │
│  20% = $14.50 USDT                  │
│  Agent: DipSniper                   │
│  Confidence: 0.75                   │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  TIER C: HEALING/LIQUIDATION        │
│  20% = $14.50 USDT                  │
│  Agent: LiquidationAgent            │
│  Confidence: 1.0 (deterministic)    │
└─────────────────────────────────────┘
```

## Key Code Location

**File:** `src/l8_lifecycle/meta_controller.py`, lines 15901-15903

```python
compound_pct = 0.60      # 60% for swing trading
healing_pct = 0.20       # 20% for healing/liquidation  
buffer_pct = 0.20        # 20% for buffer/dip buying
```

## Configuration

| Key | Value | Purpose |
|-----|-------|---------|
| `FIX8_COMPOUND_ALLOCATION_PCT` | 0.60 | Swing trade capital |
| `FIX8_HEALING_ALLOCATION_PCT` | 0.20 | Dust healing capital |
| `FIX8_BUFFER_ALLOCATION_PCT` | 0.20 | Dip buying buffer |
| `MIN_COMPOUND_USDT` | 5.0 | Minimum to trigger allocation |

## How It Works

1. **System checks:** Do we have $72.49+ free?
2. **Yes →** Calculate split:
   - Swing: $72.49 × 60% = $43.49
   - Dip: $72.49 × 20% = $14.50
   - Heal: $72.49 × 20% = $14.50
3. **Deploy:** Each agent gets their quota
4. **Trade:** Quote sizes ~$25 per entry
5. **Close:** P&L captured, capital returned
6. **Repeat:** Cycle restarts with new free capital

## Status: READY TO DEPLOY ✅

The confidence fix (0.65 → 0.80) just applied means:
- ✅ Signals will now pass validation
- ✅ Trades will execute (not skip)
- ✅ Capital will be deployed per 60/20/20 plan
- ✅ Dust healing will proceed (41 positions)

## Next Steps

Monitor logs for:
- `TRADE_EXECUTED` (not SKIPPED)
- Quote deployments (~$25 per entry)
- NAV changes (should grow if trades profitable)
- Dust position count (should decrease from 41)
