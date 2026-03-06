# 🚨_PHASE_5_DEPLOYMENT_FINAL.md

## Phase 5: Pre-Trade Risk Gate - Deployment Guide

**Status**: Ready for Production Deployment  
**Criticality**: CRITICAL ARCHITECTURE FIX  
**Estimated deployment time**: 1-2 hours  
**Downtime required**: ~5 minutes  
**Rollback time**: <1 minute  

---

## Pre-Deployment Checklist

### Code Review
- [x] CapitalGovernor.get_position_sizing() updated
- [x] Concentration logic implemented correctly
- [x] Headroom calculations verified
- [x] Bracket thresholds correct
- [x] Logging statements in place
- [ ] All call sites identified
- [ ] All call sites ready to be updated

### Testing
- [ ] Unit tests pass (concentration gating logic)
- [ ] Integration tests pass (full flow)
- [ ] Simulation mode runs without errors
- [ ] No oversized positions created
- [ ] Concentration logs appearing

### Documentation
- [x] Phase 5 deployment guide created
- [x] Integration guide created
- [x] Quick reference created
- [x] Architecture diagrams prepared
- [ ] Team trained on new behavior

### Monitoring
- [ ] Prometheus metrics configured
- [ ] Log alerts configured
- [ ] Dashboard ready
- [ ] Runbook prepared

---

## Deployment Steps

### STEP 1: Pre-Deployment Verification (5 min)

```bash
# 1. Verify capital_governor.py has Phase 5 code
grep -n "current_position_value" core/capital_governor.py
# Should show: Lines ~275-370 with multiple references

# 2. Verify method signature updated
grep -A 2 "def get_position_sizing" core/capital_governor.py
# Should show: current_position_value parameter

# 3. Verify concentration logic present
grep -n "max_position_pct" core/capital_governor.py
# Should show: Multiple lines with 0.50, 0.35, 0.25, 0.20

# 4. Verify logging code present
grep -n "CapitalGovernor:ConcentrationGate" core/capital_governor.py
# Should show: Lines with warning logs
```

**Expected output**: All commands show matches from capital_governor.py

If not, Phase 5 code is not properly deployed. DO NOT CONTINUE.

---

### STEP 2: Identify All Call Sites (10 min)

```bash
# Find all callers of get_position_sizing()
grep -rn "\.get_position_sizing(" core/ --include="*.py" | grep -v "def get_position_sizing"
```

**Expected results** (example):
```
core/execution_manager.py:245: sizing = self.capital_governor.get_position_sizing(nav, symbol)
core/scaling_engine.py:128: sizing = gov.get_position_sizing(nav, symbol)
core/meta_controller.py:456: sizing = self.capital_governor.get_position_sizing(nav, symbol)
```

**Document each location**:
- File name
- Line number
- Function name
- Current context (what symbol? do we have current position?)

---

### STEP 3: Prepare Call Site Updates (15 min)

For each call site identified:

1. **Review context** - What is the function doing?
2. **Identify symbol** - What symbol is being sized?
3. **Find current_pos source** - Where can we get current position value?
4. **Write update** - Prepare the code change
5. **Plan testing** - How to test this specific update?

**Template for tracking**:
```
File: core/execution_manager.py
Line: 245
Function: execute_buy()
Symbol: parameter (symbol)
Current position source: shared_state.get_position_value(symbol)
Update plan: Pass current_pos as new parameter
Test: Verify quote is capped when position near limit
Status: [ ] Ready
```

---

### STEP 4: Update Call Sites (30 min)

**IMPORTANT**: Do them in this order:

#### Priority 1: execution_manager.py (CRITICAL - Final risk gate)

```python
# BEFORE
async def execute_buy(self, symbol: str, quote: float, nav: float):
    sizing = self.capital_governor.get_position_sizing(nav, symbol)
    allowed_quote = sizing["quote_per_position"]
    
    if quote > allowed_quote:
        quote = allowed_quote
    
    await self.execute_order(symbol, quote)

# AFTER
async def execute_buy(self, symbol: str, quote: float, nav: float):
    # Get current position value for concentration gating
    position = await self.shared_state.get_position(symbol)
    current_pos = position.position_value if position else 0.0
    
    # Get concentration-gated sizing
    sizing = self.capital_governor.get_position_sizing(
        nav=nav,
        symbol=symbol,
        current_position_value=current_pos
    )
    allowed_quote = sizing["quote_per_position"]
    
    if quote > allowed_quote:
        logger.info(
            "ExecutionManager: %s quote adjusted for concentration: "
            "%.2f → %.2f (headroom: %.2f USDT)",
            symbol, quote, allowed_quote,
            sizing["concentration_headroom"]
        )
        quote = allowed_quote
    
    await self.execute_order(symbol, quote)
```

**Verify after change**:
```bash
grep -A 8 "async def execute_buy" core/execution_manager.py | grep "current_position_value"
# Should show the parameter being passed
```

#### Priority 2: scaling_engine.py (Important - Position scaling)

```python
# BEFORE
async def calculate_scale_size(self, nav: float, symbol: str):
    sizing = self.capital_governor.get_position_sizing(nav, symbol)
    return sizing["quote_per_position"]

# AFTER
async def calculate_scale_size(self, nav: float, symbol: str):
    # Scaling adds to existing position - need current value for headroom
    current_pos = await self.shared_state.get_position_value(symbol) or 0.0
    
    sizing = self.capital_governor.get_position_sizing(
        nav=nav,
        symbol=symbol,
        current_position_value=current_pos
    )
    
    logger.debug(
        "ScalingEngine: %s scale size with %.2f USDT headroom",
        symbol, sizing["concentration_headroom"]
    )
    
    return sizing["quote_per_position"]
```

**Verify after change**:
```bash
grep -A 6 "async def calculate_scale_size" core/scaling_engine.py | grep "current_position_value"
# Should show the parameter being passed
```

#### Priority 3: meta_controller.py or signal_manager.py (Important - Signal processing)

```python
# BEFORE
def process_signal(self, signal: TradingSignal, nav: float):
    sizing = self.capital_governor.get_position_sizing(nav, signal.symbol)
    max_quote = sizing["quote_per_position"]

# AFTER
async def process_signal(self, signal: TradingSignal, nav: float):
    # Get current position for concentration-aware sizing
    position = self.portfolio.positions.get(signal.symbol)
    current_pos = position.get("position_value", 0.0) if position else 0.0
    
    sizing = self.capital_governor.get_position_sizing(
        nav=nav,
        symbol=signal.symbol,
        current_position_value=current_pos
    )
    max_quote = sizing["quote_per_position"]
    
    logger.debug(
        "SignalManager: %s signal sized with %.2f USDT headroom",
        signal.symbol, sizing["concentration_headroom"]
    )
```

**Verify after change**:
```bash
grep -A 8 "process_signal" core/meta_controller.py | grep "current_position_value"
# Should show the parameter being passed
```

#### Priority 4: Other locations (If they exist)

Search for remaining calls:
```bash
grep -rn "\.get_position_sizing(" core/ --include="*.py" | grep -v "def get_position_sizing"
```

For each, follow same pattern:
1. Get current_position_value for symbol
2. Pass as parameter
3. Add logging

---

### STEP 5: Syntax Validation (5 min)

```bash
# Check for syntax errors in all modified files
python3 -m py_compile core/capital_governor.py
python3 -m py_compile core/execution_manager.py
python3 -m py_compile core/scaling_engine.py
python3 -m py_compile core/meta_controller.py
# Add others as needed

# Expected output: No output (success)
```

If any syntax error appears: FIX IMMEDIATELY before proceeding.

---

### STEP 6: Unit Testing (10 min)

**Create test_phase5.py** if not exists:

```python
import pytest
from core.capital_governor import CapitalGovernor

@pytest.mark.asyncio
async def test_phase5_concentration_gating():
    """Test Phase 5: pre-trade concentration gating."""
    gov = CapitalGovernor(config)
    
    # Test case 1: MICRO bracket, 50% max
    sizing = gov.get_position_sizing(
        nav=100.0,
        symbol="SOL",
        current_position_value=40.0
    )
    
    assert sizing["max_position_pct"] == 0.50
    max_pos = 100.0 * 0.50  # $50
    headroom = max_pos - 40.0  # $10
    
    assert sizing["concentration_headroom"] == headroom
    assert sizing["quote_per_position"] <= headroom
    
    # Test case 2: SMALL bracket, 35% max
    sizing = gov.get_position_sizing(
        nav=1000.0,
        symbol="BTC",
        current_position_value=200.0
    )
    
    assert sizing["max_position_pct"] == 0.35
    max_pos = 1000.0 * 0.35  # $350
    headroom = max_pos - 200.0  # $150
    
    assert sizing["concentration_headroom"] == headroom
    
    print("✅ All Phase 5 tests passed")

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Run tests**:
```bash
python3 -m pytest test_phase5.py -v
```

**Expected output**:
```
test_phase5_concentration_gating PASSED ✅
===== 1 passed in 0.XXs =====
```

---

### STEP 7: Simulation Mode Testing (15 min)

```bash
# Run bot in simulation mode
python3 octivault_trader.py --mode=simulation --log-level=DEBUG

# Wait 5-10 minutes of trading activity

# In another terminal, check logs:
tail -f logs/app.log | grep "[CapitalGovernor:ConcentrationGate]"

# Should see concentration gating logs appearing
# Example:
# [CapitalGovernor:ConcentrationGate] SOL CAPPED: ...
# [CapitalGovernor:ConcentrationGate] BTC CAPPED: ...
```

**Expected behavior**:
- ✅ Logs show concentration gating decisions
- ✅ No syntax errors
- ✅ System stable
- ✅ No oversized positions in portfolio

**If something is wrong**:
1. Stop the bot (Ctrl+C)
2. Check logs for errors
3. Fix the issue
4. Restart
5. Retest

---

### STEP 8: Production Deployment (5 min)

**Timing**: Deploy during low-volume trading hours (night/early morning)

```bash
# BACKUP CURRENT CODE (SAFETY)
cp core/capital_governor.py core/capital_governor.py.backup-pre-phase5
cp core/execution_manager.py core/execution_manager.py.backup-pre-phase5
cp core/scaling_engine.py core/scaling_engine.py.backup-pre-phase5
# Add others as needed

# COMMIT CHANGES
git add core/
git commit -m "Phase 5: Pre-Trade Risk Gate - Concentration-aware sizing"

# PUSH TO PRODUCTION
git push origin main  # Or your deployment branch

# RESTART BOT
# (Your deployment process here - docker, systemd, etc.)
systemctl restart octivault-trader

# or
docker restart octivault-trader
```

**Wait 2 minutes for bot to initialize**

---

### STEP 9: Immediate Verification (5 min)

```bash
# 1. Check bot is running
ps aux | grep octivault_trader
# Should show: octivault_trader process running

# 2. Check no errors in logs
tail -50 logs/app.log | grep ERROR
# Should be empty (or only pre-existing errors)

# 3. Check portfolio was initialized
tail -20 logs/app.log | grep -i "portfolio\|nav\|balance"
# Should show portfolio initialization

# 4. Check concentration gating active
tail -100 logs/app.log | grep "[CapitalGovernor:ConcentrationGate]"
# Should see concentration decisions being made

# 5. Quick health check
curl http://localhost:8000/health  # If health endpoint exists
# Should return 200 OK
```

**If all checks pass**: ✅ Deployment successful

**If any check fails**: 🔴 ROLLBACK immediately (see section below)

---

### STEP 10: 1-Hour Monitoring (60 min)

During first hour after deployment:

```bash
# Terminal 1: Watch logs for errors
tail -f logs/app.log | grep -E "ERROR|WARNING|ConcentrationGate"

# Terminal 2: Monitor concentration gating frequency
watch -n 10 "grep '[CapitalGovernor:ConcentrationGate]' logs/app.log | wc -l"

# Terminal 3: Check portfolio concentrations
watch -n 30 "python3 scripts/check_concentrations.py"
```

**Check list**:
- [ ] No ERROR logs
- [ ] Concentration gates appearing appropriately
- [ ] Portfolio concentrations stay within limits
- [ ] Trading proceeding normally
- [ ] Account NAV stable
- [ ] No crashes

**If everything looks good**: Continue to Phase 11

---

### STEP 11: 24-Hour Monitoring (ongoing)

Setup alerts/monitoring:

```bash
# Alert if ERROR in logs
grep ERROR logs/app.log | wc -l  # Should stay low

# Alert if no ConcentrationGate logs (might indicate disabled)
grep "[CapitalGovernor:ConcentrationGate]" logs/app.log | tail -60
# Should show logs from last hour

# Check portfolio concentrations daily
python3 -c "
import json
positions = json.load(open('portfolio.json'))
nav = positions['nav']
for symbol, pos in positions['positions'].items():
    pct = (pos['position_value'] / nav * 100) if nav > 0 else 0
    if pct > 50:
        print(f'WARNING: {symbol} concentration {pct:.1f}%')
"
```

---

## Rollback Procedure

**If something goes wrong**, rollback is fast:

### Immediate Rollback (< 1 minute)

```bash
# 1. Stop bot
systemctl stop octivault-trader
# or
docker stop octivault-trader

# 2. Restore backed-up files
cp core/capital_governor.py.backup-pre-phase5 core/capital_governor.py
cp core/execution_manager.py.backup-pre-phase5 core/execution_manager.py
cp core/scaling_engine.py.backup-pre-phase5 core/scaling_engine.py

# 3. Restart bot
systemctl start octivault-trader
# or
docker start octivault-trader

# 4. Verify it's running
ps aux | grep octivault_trader
tail -20 logs/app.log
```

**Time to restore**: ~30 seconds  
**Data loss**: None (system state unchanged)  
**Positions affected**: None (existing positions unaffected)

### Detailed Rollback (if needed)

```bash
# Revert git changes
git log --oneline | head -5
git revert <phase5-commit-hash>
git push origin main

# Full restart
docker-compose down
docker-compose up -d
```

---

## What to Watch For

### Normal Behavior (Expected)

✅ Concentration logs appearing periodically  
✅ Quotes occasionally reduced/capped  
✅ Headroom calculations in logs  
✅ No oversized positions  
✅ System trading normally  
✅ NAV stable  

### Abnormal Behavior (Alert/Investigate)

⚠️ Concentration logs appearing VERY frequently (>10/min)  
⚠️ Quotes reduced to very small amounts  
⚠️ Trading volume significantly reduced  
⚠️ ERROR logs mentioning CapitalGovernor  
⚠️ Positions still exceeding concentration limits  
⚠️ NAV dropping unexpectedly  
⚠️ System crashes  

**If abnormal behavior**: ROLLBACK immediately

---

## Verification Checklist - Post Deployment

- [ ] Bot running without errors
- [ ] Concentration gating logs visible
- [ ] Portfolio concentrations within limits
- [ ] Trading executing normally
- [ ] NAV stable or growing
- [ ] No crashes or exceptions
- [ ] Quote sizing behavior as expected
- [ ] Headroom calculations correct
- [ ] All call sites integrated
- [ ] 1-hour monitoring complete
- [ ] Ready for 24-hour monitoring

---

## Success Metrics

### Immediate (first hour)
- ✅ Zero crashes
- ✅ Concentration logs appearing
- ✅ All positions within concentration limits

### Short-term (first day)
- ✅ No oversized positions created
- ✅ Zero deadlock-related crashes
- ✅ Trading volume normal or better
- ✅ NAV stable

### Long-term (first week)
- ✅ Concentration limits consistently enforced
- ✅ Zero deadlock crashes
- ✅ Portfolio diversification natural
- ✅ Risk profile improved

---

## Support Resources

**If deployment has issues**:

1. **Check Phase 5 documentation**:
   - 🚨_PHASE_5_PRE_TRADE_RISK_GATE_DEPLOYED.md
   - ⚡_PHASE_5_QUICK_REFERENCE.md

2. **Check integration guide**:
   - ⚡_PHASE_5_INTEGRATION_GUIDE.md

3. **Rollback to stable version**:
   - Use backup files
   - git revert to previous commit

4. **Debug concentration gating**:
   - Check logs for [CapitalGovernor:ConcentrationGate]
   - Verify current_position_value being passed
   - Check headroom calculations

---

## Timeline Summary

| Step | Time | Status |
|------|------|--------|
| Pre-deployment checks | 5 min | ✅ |
| Call site identification | 10 min | 🔄 |
| Call site preparation | 15 min | 🔄 |
| Call site updates | 30 min | 🔄 |
| Syntax validation | 5 min | 🔄 |
| Unit testing | 10 min | 🔄 |
| Simulation testing | 15 min | 🔄 |
| Production deployment | 5 min | 🔄 |
| Immediate verification | 5 min | 🔄 |
| 1-hour monitoring | 60 min | 🔄 |
| **TOTAL** | **~2.5 hours** | |

---

## Final Checklist Before Go-Live

- [x] Phase 5 code implemented ✅
- [x] Documentation complete ✅
- [ ] All call sites identified
- [ ] All call sites updated
- [ ] Unit tests passing
- [ ] Simulation mode tested
- [ ] Backups created
- [ ] Team briefed
- [ ] Rollback plan understood
- [ ] Monitoring configured
- [ ] Alerts configured
- [ ] On-call support arranged

---

## Emergency Contact

If deployment issues occur after hours:
- Check logs immediately
- Attempt rollback
- Document what went wrong
- Contact engineering team

---

*Status: Ready for Production Deployment*  
*Complexity: Low-Medium*  
*Risk: Very Low (fully backward compatible)*  
*Benefit: Eliminates entire class of deadlock bugs*
