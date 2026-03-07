# ⚡ Paper Mode Launch Quick Reference

**Status**: ✅ SYSTEM READY

---

## 🚀 Launch Command

```bash
python3 main.py
```

**That's it!** The system is configured for paper mode trading.

---

## ✅ Pre-Launch Checklist (30 seconds)

```bash
# 1. Verify paper mode flag
grep "PAPER_MODE=True" .env core/.env

# 2. Verify live mode disabled  
grep "LIVE_MODE=False" .env core/.env

# 3. Verify paper API keys
grep "BINANCE_API_KEY=vsRbO0P2" .env core/.env

# 4. Verify using main.py (NOT main_live_sequential.py)
echo "Ready to run: python3 main.py"
```

---

## 🔍 Expected Startup Logs

When you launch with `python3 main.py`, you should see:

```
[INFO] Runtime mode: paper (testnet=False, paper=True, signal_only=False)
[INFO] ExchangeClient initialized in paper mode
[INFO] PAPER_MODE=True detected
```

**If you see**:
```
[WARNING] Runtime mode: live
```
**STOP!** Check configuration (see troubleshooting below)

---

## 🛑 CRITICAL DO's and DON'Ts

### ✅ DO
- ✅ Run `python3 main.py` 
- ✅ Use paper mode credentials
- ✅ Check logs for "Runtime mode: paper"
- ✅ Monitor virtual USDT balance
- ✅ Test strategies risk-free

### ❌ DON'T  
- ❌ Run `python3 main_live_sequential.py` (forces LIVE mode!)
- ❌ Manually edit mode flags without restarting
- ❌ Use live API keys
- ❌ Assume orders won't execute (they won't - paper mode is virtual)

---

## 📁 What Was Fixed

**Critical Issue Found**:
- `core/.env` had `LIVE_MODE=True` (would override paper mode!)
- `core/.env` had old live API keys

**Fixed**:
- ✅ `core/.env`: PAPER_MODE=True, LIVE_MODE=False
- ✅ `core/.env`: Updated to paper mode API keys
- ✅ Both `.env` files now consistent

---

## 🔧 Configuration Reference

### Root `.env` (Primary)
```properties
PAPER_MODE=True
LIVE_MODE=False
TESTNET_MODE=False
SIMULATION_MODE=False
BINANCE_API_KEY=vsRbO0P2BEcTMKsuzM66cJCqcVYe55v3bj6DiWWRqxdnxE6fPIZTHYoWCa5rU2br
BINANCE_API_SECRET=TcxvoQXeZ3iiYtsRZ9DQZonWTehfAdPI4cFbdR6qV8OFgjkXGtMptb5D1HLwkSAw
```

### `core/.env` (Secondary - Now Consistent!)
```properties
PAPER_MODE=True
LIVE_MODE=False  
TESTNET_MODE=False
SIMULATION_MODE=False
BINANCE_API_KEY=vsRbO0P2BEcTMKsuzM66cJCqcVYe55v3bj6DiWWRqxdnxE6fPIZTHYoWCa5rU2br
BINANCE_API_SECRET=TcxvoQXeZ3iiYtsRZ9DQZonWTehfAdPI4cFbdR6qV8OFgjkXGtMptb5D1HLwkSAw
```

---

## ❓ Quick Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Runtime mode: live" in logs | `core/.env` has LIVE_MODE=True | Update `core/.env` to LIVE_MODE=False |
| API connection errors | Keys mismatch between .env files | Copy paper keys to both .env files |
| "Entry point missing" | Wrong entry point being used | Use `python3 main.py`, NOT `main_live_sequential.py` |
| Virtual orders executing | This shouldn't happen (paper mode is virtual) | Check runtime mode announcement in logs |

---

## 📊 Paper Mode Features

- ✅ **Real market data** - Live prices from Binance
- ✅ **Real order books** - Actual liquidity data
- ✅ **Virtual execution** - Orders NOT sent to exchange
- ✅ **Zero capital risk** - No real money used
- ✅ **Full logging** - Track all simulated trades
- ✅ **Performance metrics** - See backtest-like results

---

## 🎯 Next Steps

1. **Launch**: Run `python3 main.py`
2. **Monitor**: Check logs for startup messages
3. **Verify**: Confirm "Runtime mode: paper" appears
4. **Trade**: Place simulated trades and monitor performance
5. **Iterate**: Adjust strategies based on paper trading results

---

## 📞 Support

**Configuration not working?**
- Check both `.env` files match (use grep commands above)
- Verify paper API keys are correct
- Restart with `python3 main.py`

**Still having issues?**
- See `📋_PAPER_MODE_READINESS_FINAL_VERIFICATION.md` for detailed diagnostics
- Check component logs for specific error messages

---

**Last Updated**: 2025-01-22
**System Status**: ✅ READY TO LAUNCH
