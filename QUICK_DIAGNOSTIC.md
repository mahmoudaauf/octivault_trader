# 🔍 Signal Filtering Diagnostics - Quick Reference

## TL;DR - Run This

```bash
# Run bot with diagnostics
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python main_phased.py 2>&1 | tee logs/diagnostic_run.log

# Quick analysis (after bot runs for 1-2 minutes)
python analyze_diagnostics.py logs/diagnostic_run.log
```

## What You'll See

| Log Message | Meaning | What To Look For |
|-------------|---------|-----------------|
| `[Meta:SIGNAL_INTAKE]` | Signals from cache | Count > 0 means TrendHunter is working |
| `[Meta:GATE_TRACE]` | Signal enters filter | Should match SIGNAL_INTAKE count |
| `[Meta:GATE_DROP_TRADEABILITY]` | Confidence too low | Most common reason for signal drops |
| `[Meta:GATE_DROP_RECOVERY]` | In capital recovery | Capital recovery suppressing BUYs |
| `[Meta:GATE_DROP_ONE_POSITION]` | Position exists | Symbol already has open trade |
| `[Meta:GATE_PASSED]` | Signal approved! | Great! Signal will become decision |
| `[Meta:AFTER_FILTER]` | Final signal count | 0 = all dropped, >0 = some made it |
| `[Meta:DEADLOCK_DIAGNOSTIC]` | Why nothing passed | Error explaining the problem |

## Common Problems & Fixes

### Problem: "CONFIDENCE TOO LOW"
```
[Meta:GATE_DROP_TRADEABILITY] OPNUSDT BUY dropped: conf=0.45 floor=0.55
```
**Fix Options:**
1. Check market conditions - TrendHunter may need better signals
2. Check bot mode - might be in RECOVERY with elevated floor
3. Adjust TrendHunter confidence calculation
4. Lower confidence floor in MetaController settings

### Problem: "CAPITAL RECOVERY SUPPRESSING"
```
[Meta:GATE_DROP_RECOVERY] OPNUSDT BUY dropped at CAPITAL_RECOVERY gate
```
**Fix Options:**
1. Exit capital recovery mode: check shared_state settings
2. Wait for recovery to complete naturally
3. Generate bootstrap signals to escape FLAT state

### Problem: "NO SIGNALS GENERATED"
```
[Meta:SIGNAL_INTAKE] Retrieved 0 signals from cache: []
```
**Fix Options:**
1. Check TrendHunter logs - agent may be erroring
2. Verify market data is flowing (check OHLCV cache)
3. Restart TrendHunter agent
4. Check agent configuration matches portfolio

### Problem: "ALL SIGNALS DROPPED"
```
[Meta:GATE_TRACE] Processing OPNUSDT BUY from TrendHunter (conf=0.77)
[Meta:GATE_DROP_TRADEABILITY] OPNUSDT BUY dropped at TRADEABILITY gate: conf=0.77 floor=0.85
[Meta:AFTER_FILTER] valid_signals_by_symbol has 0 symbols
```
**Root Cause Identification:**
- Look at the GATE_DROP message to see which gate is dropping ALL signals
- If TRADEABILITY drops all: confidence floor is too high
- If RECOVERY drops all: capital recovery mode is active
- If ONE_POSITION drops all: too many symbols have positions

## Diagnostic Checks

### Check 1: Is TrendHunter Working?
```bash
grep "\[Meta:SIGNAL_INTAKE\]" logs/diagnostic_run.log | tail -5
# Should show: "Retrieved 3+ signals from cache"
```

### Check 2: Which Gate Drops Most?
```bash
grep "\[Meta:GATE_DROP" logs/diagnostic_run.log | \
  sed 's/.*GATE_DROP_//' | \
  sed 's/\].*//' | \
  sort | uniq -c | sort -rn
```
**Example output:**
```
5 TRADEABILITY  ← This gate is dropping signals!
2 RECOVERY
1 ONE_POSITION
```

### Check 3: Do Any Signals Pass?
```bash
grep "\[Meta:GATE_PASSED\]" logs/diagnostic_run.log | wc -l
# If 0: ALL signals are dropped
# If >0: Some signals ARE passing, check execution logs
```

### Check 4: What's the Deadlock Reason?
```bash
grep "\[Meta:DEADLOCK_DIAGNOSTIC\]" logs/diagnostic_run.log
# Explains which gate(s) are responsible
```

## Signal Flow Visualization

```
TrendHunter generates signal
           ↓
[Meta:SIGNAL_INTAKE] logs it
           ↓
For each signal:
  [Meta:GATE_TRACE] logs entry
           ↓
  Capital Recovery gate?
  ├─ DROP? → [Meta:GATE_DROP_RECOVERY]
  ├─ PASS → continue
           ↓
  Confidence floor gate?
  ├─ DROP? → [Meta:GATE_DROP_TRADEABILITY]
  ├─ PASS → continue
           ↓
  One-position gate?
  ├─ DROP? → [Meta:GATE_DROP_ONE_POSITION]
  ├─ PASS → continue
           ↓
  Other gates (many more)
           ↓
  PASSED ALL? → [Meta:GATE_PASSED]
           ↓
Signal added to valid_signals_by_symbol
           ↓
[Meta:AFTER_FILTER] shows final count
```

## Read This If You Want Details

- **DIAGNOSTIC_GUIDE.md** - Full diagnostic guide with scenarios
- **SIGNAL_FILTER_DIAGNOSTICS_README.md** - Implementation details
- **analyze_diagnostics.py** - Auto-analysis script

## Run Diagnostics Now!

```bash
# Terminal 1: Run the bot
cd ~/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python main_phased.py 2>&1 | tee logs/diagnostic_run.log

# Wait 1-2 minutes for bot to loop several times...

# Terminal 2: Analyze the logs
cd ~/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python analyze_diagnostics.py logs/diagnostic_run.log
```

The diagnostic output will tell you exactly where signals are being dropped and why! 🎯
