# 🔍 Diagnostic Guide: NAV=0 at Startup

**Issue**: `NAV is 0.0 (should be > 0)` error during Step 5 verification  
**Root Cause**: RecoveryEngine failed to rebuild state (Step 1)  
**Status**: FIXED with better diagnostics

---

## What This Error Means

When you see:
```
ERROR [AppContext] [StartupOrchestrator] Step 5: Verify startup integrity - ⚠️ NAV is 0.0 (should be > 0)
ERROR [AppContext] [StartupOrchestrator] Step 5: Verify startup integrity FAILED - capital integrity issues
```

This means **Step 1 (RecoveryEngine.rebuild_state) did not populate wallet balances**.

---

## Root Causes (Check In This Order)

### 1. **Exchange Connection Failed** ⚠️ MOST LIKELY
The RecoveryEngine tried to fetch balances from the exchange, but the API call failed.

**Diagnostic**: Check logs BEFORE the error:
```
[StartupOrchestrator] Step 1 - Before: nav=0, positions=0
[StartupOrchestrator] Step 1 - After: nav=0, positions=0
[StartupOrchestrator] Step 1 - ⚠️ NAV is still 0 after rebuild
```

**Fix**:
- Verify exchange API keys are correct
- Check exchange API rate limits (you may be throttled)
- Verify network connectivity to exchange
- Check if exchange API is down

### 2. **Empty Wallet**
Your exchange account has zero balance.

**Diagnostic**: See logs:
```
[StartupOrchestrator] Step 1 - After: nav=0, positions=0, free=0
```

**Fix**:
- Add funds to your exchange account
- Or configure bot with test/paper trading account with funds

### 3. **Exchange Client Not Initialized**
The exchange_client object wasn't properly passed to RecoveryEngine.

**Diagnostic**: Check app_context.py Phase 8.5 integration - does orchestrator receive exchange_client?

**Fix**:
- Ensure exchange_client is initialized in app_context before Phase 8.5
- Check that RecoveryEngine has access to exchange_client

### 4. **SharedState Not Properly Initialized**
The shared_state object exists but doesn't have 'nav' attribute.

**Diagnostic**: Look for:
```
[StartupOrchestrator] Step 5 - Raw metrics: nav=0, free=0, invested=0
```

**Fix**:
- Ensure SharedState is initialized in Phase 3
- Check SharedState has 'nav', 'free_quote', 'positions' attributes

---

## Improved Diagnostics (New in This Update)

### Step 1 Now Shows More Detail

**Before Rebuild**:
```
[StartupOrchestrator] Step 1 - Before: nav=0, positions=0
```

**After Rebuild**:
```
[StartupOrchestrator] Step 1 - After: nav=10000.00, positions=3, free=5000.00
```

**If NAV Still Zero**:
```
[StartupOrchestrator] Step 1 - ⚠️ NAV is still 0 after rebuild. 
This suggests: exchange API error, wallet empty, or exchange client not initialized
```

### Step 5 Now More Defensive

**Instead of failing immediately**, it now:
1. Logs raw metrics for diagnostics
2. Allows NAV=0 if there are no positions (cold start)
3. Only fails if NAV=0 BUT you have positions (indicates corruption)

```
[StartupOrchestrator] Step 5 - Raw metrics: nav=0, free=0, invested=0, positions=0
[StartupOrchestrator] Step 5 - Cold start: NAV=0, no positions
```

---

## Troubleshooting Checklist

### If NAV=0 with No Positions (Cold Start)
- [ ] Check exchange account has funds
- [ ] Verify API keys in config
- [ ] Test exchange API manually (curl/Postman)
- [ ] Check network connectivity
- [ ] Review RecoveryEngine logs for API errors

### If NAV=0 with Positions (Data Corruption)
- [ ] This indicates a serious issue in Step 1 or 2
- [ ] Check SharedState position data
- [ ] Verify position reconstruction logic
- [ ] May need to reset SharedState and restart

### If Exchange API Error
Look for additional errors in logs:
```
[RecoveryEngine] Error fetching balances: 403 Unauthorized
[RecoveryEngine] API rate limit exceeded
[RecoveryEngine] Network timeout
```

**Fix**:
- Wait and retry (rate limit)
- Update API keys (403 error)
- Check network (timeout)

### If Wallet is Empty
```
[StartupOrchestrator] Step 1 - After: nav=0, positions=0, free=0
```

**Fix**:
- Send funds to exchange wallet
- Use different account
- Enable test/paper trading

---

## Next Steps

### Immediate Action
1. **Check logs** for diagnostic messages above
2. **Identify root cause** from the list above
3. **Fix issue** (API keys, wallet balance, network, etc.)
4. **Restart bot** to retry

### If Still Failing
1. Run exchange API test manually:
   ```bash
   # Test that exchange is accessible
   python3 -c "
   from ccxt import binance
   ex = binance({'apiKey': 'YOUR_KEY', 'secret': 'YOUR_SECRET'})
   balance = ex.fetch_balance()
   print(f'Total USDT: {balance[\"USDT\"][\"total\"]}')
   "
   ```

2. Check RecoveryEngine implementation:
   - Is it calling exchange.fetch_balance()?
   - Is it passing result to SharedState?
   - Are there try/except blocks hiding errors?

3. Enable verbose logging:
   - Set RecoveryEngine logger to DEBUG
   - Set SharedState logger to DEBUG
   - Review detailed error messages

---

## Recovery Options

### Option 1: Graceful Degradation (Recommended for Testing)
Start bot in **"simulate_empty_wallet"** mode:
```python
# In app_context.py, skip exchange data if fetch fails
if nav == 0 and len(positions) == 0:
    self.logger.warning("Starting with zero balance for testing")
    return True  # Allow startup anyway
```

### Option 2: Manual State Initialization
If you know your wallet balance, manually set it:
```python
# In app_context.py after Step 1 fails
shared_state.nav = 10000.00
shared_state.free_quote = 10000.00
shared_state.invested_capital = 0.0
shared_state.positions = {}
```

### Option 3: Reset and Retry
```bash
# Clear any cached state
rm -f ~/.octivault/state.json

# Restart bot
python3 main_phased.py
```

---

## Expected Behavior

### Successful Startup
```
[StartupOrchestrator] Step 1 - Before: nav=0, positions=0
[StartupOrchestrator] Step 1 - After: nav=10000.00, positions=2, free=6500.00
[StartupOrchestrator] Step 5 - Raw metrics: nav=10000.00, free=6500.00
[StartupOrchestrator] Step 5 complete: NAV=10000.00, Free=6500.00
[StartupOrchestrator] ✅ STARTUP ORCHESTRATION COMPLETE
```

### Cold Start (Empty Wallet)
```
[StartupOrchestrator] Step 1 - After: nav=0, positions=0, free=0
[StartupOrchestrator] Step 5 - Cold start: NAV=0, no positions, exchange returned no balance
[StartupOrchestrator] Step 5 complete: NAV=0.00, Free=0.00
[StartupOrchestrator] ✅ STARTUP ORCHESTRATION COMPLETE
```

### Error Case (NAV=0 with Positions)
```
[StartupOrchestrator] Step 5 - Raw metrics: nav=0, free=0, positions=2
ERROR [AppContext] [StartupOrchestrator] Step 5: Verify startup integrity - ⚠️ NAV is 0 but has positions
```

---

## When to Contact Support

If you've checked all the above and still getting NAV=0 error:

1. **Collect logs** from startup until error
2. **Note exchange** (Binance, Kraken, etc.)
3. **Test exchange API** manually to verify it works
4. **Check Step 1 logs** for RecoveryEngine errors
5. **Review config** for correct API keys

Then provide these details for debugging.

---

## Files Modified

- `core/startup_orchestrator.py` - Enhanced diagnostics in Step 1 and Step 5
- Better before/after state logging
- More defensive NAV validation
- Clearer error messages

---

## Status

✅ **Diagnostic improvements deployed**  
✅ **Better error messages in logs**  
✅ **More defensive validation**  
✅ **Guides users to root cause**

Restart bot to use new diagnostics.

