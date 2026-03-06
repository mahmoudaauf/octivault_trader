# 🔄 PROPOSAL UNIVERSE ADDITION: ARCHITECTURE DIAGRAM

## System Flow (After Fix)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROPOSAL UNIVERSE FLOW                          │
└─────────────────────────────────────────────────────────────────────────┘

DISCOVERY PHASE (Multiple Passes)
═════════════════════════════════════════════════════════════════════════

Pass 1: SymbolScreener Scan
  ┌──────────────────────────┐
  │ Exchange 24h Tickers     │
  │ 1000+ symbols available  │
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │ Filter & Score           │
  │ - Volatility (ATR%)      │
  │ - Volume (24h)           │
  │ - Liquidity              │
  └────────────┬─────────────┘
               │
               ├─► Filtered: [BTCUSDT, ETHUSDT, BNBUSDT, ...]
               │
               ▼
  ┌──────────────────────────────────────────┐
  │ propose_symbols() → SymbolManager         │
  │ merge_mode=True (ADDITIVE)                │
  └────────────┬─────────────────────────────┘
               │
               ▼
  SHARED STATE (Before): []
           │
           ├─► MERGE incoming [BTCUSDT, ETHUSDT, BNBUSDT] 
           │   with current []
           │
           ▼
  SHARED STATE (After):  [BTCUSDT, ETHUSDT, BNBUSDT]


Pass 2: SymbolScreener Scan (30 mins later)
  ┌──────────────────────────┐
  │ Exchange 24h Tickers     │
  │ 1000+ symbols available  │
  │ (Market conditions changed)
  └────────────┬─────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │ Filter & Score           │
  │ (Different ATR%, Volume) │
  └────────────┬─────────────┘
               │
               ├─► Filtered: [ADAUSDT, XRPUSDT, DOGEUSDT, ...]
               │
               ▼
  ┌──────────────────────────────────────────┐
  │ propose_symbols() → SymbolManager         │
  │ merge_mode=True (ADDITIVE)                │
  └────────────┬─────────────────────────────┘
               │
               ▼
  SHARED STATE (Before): [BTCUSDT, ETHUSDT, BNBUSDT]
           │
           ├─► MERGE incoming [ADAUSDT, XRPUSDT, DOGEUSDT] 
           │   with current [BTCUSDT, ETHUSDT, BNBUSDT]
           │
           ▼
  SHARED STATE (After):  [BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, XRPUSDT, DOGEUSDT]


Pass 3: IPOChaser Agent
  ┌──────────────────────────┐
  │ New Listing Detection     │
  └────────────┬─────────────┘
               │
               ├─► Found: [NEWTOKEN1USDT, NEWTOKEN2USDT]
               │
               ▼
  ┌──────────────────────────────────────────┐
  │ propose_symbols() → SymbolManager         │
  │ merge_mode=True (ADDITIVE)                │
  └────────────┬─────────────────────────────┘
               │
               ▼
  SHARED STATE (Before): [BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, XRPUSDT, DOGEUSDT]
           │
           ├─► MERGE incoming [NEWTOKEN1USDT, NEWTOKEN2USDT]
           │   with current
           │
           ▼
  SHARED STATE (After):  [BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, XRPUSDT, DOGEUSDT, 
                          NEWTOKEN1USDT, NEWTOKEN2USDT]


CAP ENFORCEMENT
═════════════════════════════════════════════════════════════════════════

If Cap = 50 symbols:

  After Pass 3:
    Universe Size = 8 symbols  ✅ Under cap (8 < 50)
    
  After Pass 10 (continued discovery):
    Universe Size = 55 symbols  ⚠️ Over cap (55 > 50)
    
    ▼ Apply Cap (keep top 50 by priority)
    
    Universe Size = 50 symbols  ✅ Capped


COMPARISON: BEFORE vs AFTER FIX
═════════════════════════════════════════════════════════════════════════

BEFORE FIX (REPLACE MODE):
───────────────────────────

Pass 1: propose_symbols([BTCUSDT, ETHUSDT, BNBUSDT])
  ▼
  Universe: [BTCUSDT, ETHUSDT, BNBUSDT]

Pass 2: propose_symbols([ADAUSDT, XRPUSDT, DOGEUSDT])
  ▼
  Universe: [ADAUSDT, XRPUSDT, DOGEUSDT]  ❌ Lost BTCUSDT, ETHUSDT, BNBUSDT


AFTER FIX (MERGE MODE):
──────────────────────

Pass 1: propose_symbols([BTCUSDT, ETHUSDT, BNBUSDT])
  ▼
  Universe: [BTCUSDT, ETHUSDT, BNBUSDT]

Pass 2: propose_symbols([ADAUSDT, XRPUSDT, DOGEUSDT])
  ▼
  Universe: [BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, XRPUSDT, DOGEUSDT]  ✅ Kept all!


CODE EXECUTION FLOW
═════════════════════════════════════════════════════════════════════════

SymbolScreener._process_and_add_symbols([candidates])
  │
  ├─► for each candidate:
  │     │
  │     ▼
  │     SymbolManager.propose_symbol(symbol, source="SymbolScreener")
  │     │
  │     ▼
  │     SymbolManager.add_symbol(symbol, source="SymbolScreener")
  │     │
  │     ├─► Get current snapshot: await _get_symbols_snapshot()
  │     │
  │     ├─► Build final_map with all symbols:
  │     │   final_map = dict(current_snapshot)
  │     │   final_map[symbol] = metadata
  │     │
  │     ├─► Apply cap if needed
  │     │
  │     ▼
  │     SymbolManager._safe_set_accepted_symbols(
  │         final_map, 
  │         allow_shrink=False,
  │         merge_mode=True,  ◄──── KEY CHANGE!
  │         source="SymbolScreener"
  │     )
  │     │
  │     ▼
  │     SharedState.set_accepted_symbols(
  │         sanitized_map,
  │         allow_shrink=False,
  │         merge_mode=True,  ◄──── KEY CHANGE!
  │         source="SymbolScreener"
  │     )
  │     │
  │     ▼
  │     Inside SharedState.set_accepted_symbols():
  │     │
  │     ├─► if merge_mode:
  │     │   working_symbols = dict(self.accepted_symbols)  ← Start with CURRENT
  │     │   working_symbols.update(symbols)                 ← ADD incoming
  │     │   │
  │     │   └─► RESULT: [old] + [new] = combined
  │     │
  │     ├─► Apply cap if needed
  │     │
  │     ▼
  │     Update self.accepted_symbols = working_symbols
  │     Log: "[SS] 🔄 MERGE MODE: 2 + 1 = 3 symbols (source=SymbolScreener)"
  │
  └─► Next candidate...


USAGE MATRIX
═════════════════════════════════════════════════════════════════════════

Method                           merge_mode   allow_shrink   Use Case
─────────────────────────────────────────────────────────────────────────
initialize_symbols()             False        True           Initial universe setup
add_symbol()                      True         False          Single proposal (discovery)
propose_symbol()                  (via add)    False          Single proposal (discovery)
propose_symbols()                 True         False          Batch proposal (discovery)
flush_buffered_proposals()        False        True           Finalize after cap reached
finalize_universe()               False        param          Explicit shrink/trim
set_accepted_symbols() (public)   False        param          Legacy interface


GOVERNOR CAP ENFORCEMENT
═════════════════════════════════════════════════════════════════════════

Located at: SharedState.set_accepted_symbols() [canonical]

    Input symbols (after merge if applicable)
           │
           ▼
    Apply capital_symbol_governor.compute_symbol_cap()
           │
           ├─► If cap = 50 and input = 60
           │   ├─► Trim to 50 (keeps first 50)
           │   └─► Log: "🎛️ CANONICAL GOVERNOR: 60 → 50 symbols"
           │
           ▼
    Output to accepted_symbols (with cap applied)
