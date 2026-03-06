# 🔴 ROOT CAUSE ANALYSIS: Signal Cache Empty (0 signals)

## Observation from Logs

```
[Meta:BOOTSTRAP_DEBUG] Signal cache contains 0 signals: []
[Meta:POST_BUILD] decisions_count=0 decisions=[]
```

**This is REPEATED EVERY TICK** - consistently zero signals in cache.

## Root Cause: NOT The Direct Path

The direct path I added **DOESN'T EXECUTE** because signals **NEVER REACH IT**.

The actual issue is **UPSTREAM**:

```
❌ Agents NOT generating signals
  ↓
❌ generate_signals() returns empty
  ↓
❌ No intents to forward
  ↓
❌ Direct path has nothing to forward
  ↓
Signal cache stays empty
```

## Why Agents Aren't Generating Signals

### Possibility 1: Agents Not Registered
If `strategy_agents` list is empty in `collect_and_forward_signals()`:
- Check if agents are registered: `self.agents.keys()`
- Check if they have `agent_type == "strategy"`: `getattr(agent, "agent_type")`
- Check if they have `generate_signals` method: `hasattr(agent, "generate_signals")`

**Fix Applied**: Added logging to show registered agents and their types

### Possibility 2: generate_signals() Not Being Called
If agents are registered but `generate_signals()` is never called:
- Check if `collect_and_forward_signals()` is actually being called
- Check if the method is being skipped
- Check if there's an exception silently caught

**Fix Applied**: Added logging at the start of agent loop with agent types

### Possibility 3: generate_signals() Returns Empty List
If the method is called but returns `[]`:
- Agents iterate through symbols but find none
- Agents filter out all signals (confidence gate?)
- Agents are not finding trading conditions

**Fix Applied**: Added logging of batch size and intent details

### Possibility 4: _normalize_to_intents() Filtering Everything
If signals are generated but normalization filters them all:
- Symbol format mismatch (e.g., "BTC/USDT" vs "BTCUSDT")
- Confidence too low
- Action not "BUY"/"SELL"

**This is LEAST likely** - normalization would log warnings

## Diagnostic Strategy

The added logging will show us **EXACTLY** where the signal pipeline breaks:

### Phase 1: Agent Registration
```
[AgentManager] ⚠️ NO STRATEGY AGENTS FOUND! 
registered_agents=['MLForecaster', 'TrendHunter']
agent_types={'MLForecaster': 'strategy', 'TrendHunter': 'strategy'}
```

**If you see this**: Agents ARE registered, move to Phase 2

**If agents list is EMPTY**: Agents weren't registered → Check auto_register_agents()

### Phase 2: Signal Generation
```
[TrendHunter] Normalized X intents (scanned Y symbols)
```

**If you see this with X > 0**: Agents ARE generating → signals reach batch → check Phase 3

**If X = 0**: Agents generated 0 intents → check symbol loading

### Phase 3: Batch Submission  
```
[AgentManager:BATCH] Submitted batch of N intents: [...]
```

**If you see this**: Batch is non-empty → signals reach direct path

**If you DON'T see this**: Batch is empty → no signals generated

### Phase 4: Direct Path Execution
```
[AgentManager:DIRECT] Forwarded M signals directly to MetaController.signal_cache
```

**If you see this**: Direct path IS working → signals in cache

**If you DON'T see this**: Direct path not executing (batch empty)

## Expected Log Sequence (PER TICK)

When working correctly:
```
[AgentManager] Signal Collection Tick
[AgentManager] NO STRATEGY AGENTS? registered_agents=['TrendHunter'] ...
[TrendHunter] run_once start
[TrendHunter] Processing BTCUSDT
[TrendHunter] Signal collected: BTCUSDT BUY conf=0.75
[AgentManager] Normalized 1 intents (scanned 42 symbols)
[AgentManager:BATCH] Submitted batch of 1 intents: [TrendHunter:BTCUSDT]
[AgentManager:DIRECT] Forwarded 1 signals directly to MetaController.signal_cache
[SignalManager] Signal ACCEPTED and cached: BTCUSDT from TrendHunter
[Meta:POST_BUILD] decisions_count=1 decisions=[...]
```

When **BROKEN** (current state):
```
[AgentManager] Signal Collection Tick
[AgentManager] ⚠️ NO STRATEGY AGENTS FOUND! registered_agents=[]
[AgentManager] No TradeIntents collected this tick
[Meta:POST_BUILD] decisions_count=0 decisions=[]
```

OR:

```
[AgentManager] Signal Collection Tick
[AgentManager] NO STRATEGY AGENTS? registered_agents=['TrendHunter'] ...
[TrendHunter] run_once start
[TrendHunter] No symbols configured or fetched, skipping
[AgentManager] No TradeIntents collected this tick
[Meta:POST_BUILD] decisions_count=0 decisions=[]
```

## Next Steps

1. **Run the system again** with the added logging
2. **Grep for the diagnostic logs** I added:
   ```
   grep "NO STRATEGY AGENTS\|:BATCH\|:DIRECT\|Normalized" logs/trading.log
   ```
3. **Share the output** - it will pinpoint exactly where the signal breaks

## Most Likely Culprits (In Order)

1. **Agents not registered** → auto_register_agents() failing
2. **Agents have no symbols** → symbol loading broken
3. **Agents generating but all signals filtered** → confidence gates too high
4. **Symbol format mismatch** → normalization filtering valid signals

The added DEBUG logging will definitively show which one it is.
