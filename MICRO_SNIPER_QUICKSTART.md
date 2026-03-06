# MICRO_SNIPER Mode - Quick Start

## What Is This?

A system that automatically switches Octivault Trader into "MICRO_SNIPER" mode when your account has < $1000, simplifying the trading logic to focus on single-position precision.

## How Does It Work?

Every trading cycle, the system checks your NAV (account balance in USDT):

```
NAV < $1000         → MICRO_SNIPER mode (simplified)
$1000 - $5000       → STANDARD mode (normal)
$5000+              → MULTI_AGENT mode (full features)
```

Each mode has different rules for:
- How many positions you can have
- How many symbols you can trade
- Minimum signal quality required
- Maximum trades per day

## Quick Reference

### MICRO_SNIPER (NAV < $1000)

**Good For**: Micro-cap accounts where every trade matters

| Setting | Value | Why |
|---------|-------|-----|
| Max positions | 1 | Focus capital on best opportunity |
| Max symbols | 1 | Avoid over-diversification |
| Min expected_move | 1.0% | Reject low-edge signals |
| Min confidence | 0.70 | Quality filter |
| Max trades/day | 3 | Prevent over-trading |
| Position size | 30% NAV | Conservative risk |

**Disabled Features**:
- ❌ Rotation (can't switch between symbols)
- ❌ Dust healing (avoid micro-friction costs)
- ❌ Capital reservations (use all available)

**Example**:
```
Account: $116 USDT
Max position: $34.80 (30% of $116)
Can only trade ETHUSDT (1 symbol)
Can only have 1 open trade
Need signals with 1.0%+ expected move
Can trade max 3 times today
```

### STANDARD ($1000 - $5000)

**Good For**: Small accounts with room to scale

| Setting | Value |
|---------|-------|
| Max positions | 2 |
| Max symbols | 2-3 |
| Min expected_move | 0.50% |
| Min confidence | 0.65 |
| Max trades/day | 6 |
| Position size | 25% NAV |

**Enabled Features**:
- ✅ Rotation (switch between symbols)
- ✅ Dust healing (consolidate dust positions)
- ✅ Full capital allocation

### MULTI_AGENT ($5000+)

**Good For**: Larger accounts with full features

| Setting | Value |
|---------|-------|
| Max positions | 3+ |
| Max symbols | 5+ |
| Min expected_move | 0.30% |
| Min confidence | 0.60 |
| Max trades/day | 20+ |
| Position size | 20% NAV |

**All Features Enabled**

## Installation

1. Copy `core/nav_regime.py` to your `/core/` directory
2. Deploy modified `core/meta_controller.py`
3. System automatically uses new regime on startup

## Validation

```bash
cd /path/to/octivault_trader
python3 validate_micro_sniper.py
```

Should show:
```
✅ ALL CHECKS PASSED - Ready for deployment
```

## How to Know If It's Working

Look for log entries like:

```
[REGIME] NAV=842.33 USD → regime=MICRO_SNIPER (max_pos=1, max_symbols=1, min_move=1.00%, min_conf=0.70)
```

This appears at the start of each trading cycle and shows:
- Your current account balance (NAV)
- Which regime is active
- What constraints are in place

## Troubleshooting

### "Signal rejected - expected_move too low"

**At NAV < $1000?** You need 1.0%+ expected move signals.  
**Solution**: 
- Improve signal quality, OR
- Increase account to $1000+ to unlock 0.50% threshold

### "Daily limit reached (3/3 trades executed)"

**This is a safety feature in MICRO_SNIPER mode.**  
Wait for UTC midnight (automatic reset).

### "Can't trade BTCUSDT - max symbols reached"

**In MICRO_SNIPER mode, you can only trade 1 symbol at a time.**  
- Exit ETHUSDT position first, then enter BTCUSDT, OR
- Increase account to $1000+ to enable multi-symbol trading

### "Rotation blocked - no multi-symbol trading in MICRO"

**MICRO_SNIPER doesn't allow symbol rotation.**  
Increase account to $1000+ to unlock rotation.

## Common Questions

**Q: Why does my system behavior change when I reach $1000?**  
A: Different account sizes need different strategies. Small accounts benefit from focused simplicity; larger accounts need full feature set.

**Q: Can I disable regime switching?**  
A: Set your NAV high enough (≥$5000) to always be in MULTI_AGENT mode, or modify the `nav_regime.py` thresholds.

**Q: Do I lose money because of regime rules?**  
A: No. Regime rules are *conservative* - they prevent bad trades, not good ones. They gate on:
- Minimum signal quality (reject low-edge trades)
- Maximum trades per day (prevent over-trading)
- Position sizing limits (maintain risk discipline)

**Q: What happens if I cross $1000?**  
A: Automatic switch happens at next trading cycle. No restart needed. Features are enabled smoothly.

**Q: What's the "daily trade limit" reset time?**  
A: UTC midnight (00:00 UTC). Counter resets automatically.

**Q: Can I trade both ETHUSDT and BTCUSDT at NAV < $1000?**  
A: No. MICRO_SNIPER only allows 1 symbol at a time. Choose your best opportunity.

## Performance Impact

- **CPU**: <1ms per cycle (negligible)
- **Memory**: <2 KB (negligible)
- **Latency**: No change (gating is synchronous)

## When to Scale to STANDARD

Recommended milestones:
- Account reaches $1000 → Switch to STANDARD mode
- Account reaches $5000 → Switch to MULTI_AGENT mode

## Log Monitoring

All regime decisions are logged with `[REGIME]` prefix:

```
[REGIME] NAV=1005.42 USD → regime=STANDARD (...)
[REGIME:ExpectedMove] REJECT: move=0.30% < regime_min=0.50%
[REGIME:DailyLimit] REJECT: 3 trades executed today >= 3 (MICRO_SNIPER)
[REGIME_SWITCH] NAV=1005.42 USD: MICRO_SNIPER → STANDARD (switch_count=1)
[REGIME:TradeLogged] BUY ETHUSDT 0.0050 @ 2450.00 (quote=12.25), daily=1/3
```

## Support

For issues or questions:
1. Check `MICRO_SNIPER_MODE_INTEGRATION.md` for detailed architecture
2. Run `validate_micro_sniper.py` to verify installation
3. Review logs with `[REGIME]` prefix to understand gating decisions

---

**Version**: 1.0  
**Status**: Production-Ready  
**Last Updated**: March 2, 2026
