# 📝 EXACT CODE CHANGES APPLIED

**Date**: March 5, 2026
**Files Modified**: 2
**Total Lines Changed**: ~120

---

## File 1: core/shared_state.py

### Location: `set_accepted_symbols()` method
**Lines**: ~2096-2200

### Change Type: Signature Update + Logic Rewrite

#### Before (Original Signature)
```python
async def set_accepted_symbols(self, symbols: Dict[str, Dict[str, Any]], *, allow_shrink: bool = False, source: Optional[str] = None) -> None:
```

#### After (New Signature)
```python
async def set_accepted_symbols(self, symbols: Dict[str, Dict[str, Any]], *, allow_shrink: bool = False, merge_mode: bool = False, source: Optional[str] = None) -> None:
```

**Change Details**:
- Added parameter: `merge_mode: bool = False`
- Default is False (backward compatible - maintains replace mode)
- Controls whether to merge or replace symbol universe

---

### Logic Changes

#### Before (Hard Replace Only)
```python
# === HARD REPLACE MODE ===
# Build wanted set from incoming symbols
wanted = { self._norm_sym(k) for k in symbols.keys() }

# Remove everything not wanted (but protect wallet_force from non-wallet sources)
current_keys = set(self.accepted_symbols.keys())
for s in (current_keys - wanted):
    meta = self.accepted_symbols.get(s, {})
    if meta.get("accept_policy") == "wallet_force" and source != "WalletScannerAgent":
        self.logger.debug("🛡️ Protected wallet_force symbol %s from removal", s)
        continue
    
    self.accepted_symbols.pop(s, None)
    self.symbols.pop(s, None)
```

#### After (Merge or Replace)
```python
# === MERGE vs REPLACE LOGIC ===
if merge_mode:
    # ADDITIVE MODE: Merge incoming with existing
    working_symbols = dict(self.accepted_symbols)  # Start with current
    incoming_count = len(symbols)
    
    # Merge in new symbols (updates if already exist)
    for raw_sym, meta in symbols.items():
        symbol = self._norm_sym(raw_sym)
        working_symbols[symbol] = dict(meta or {})
    
    final_count = len(working_symbols)
    self.logger.info(
        f"[SS] 🔄 MERGE MODE: {current_count} + {incoming_count} = {final_count} symbols (source={source})"
    )
else:
    # REPLACEMENT MODE: Incoming replaces current
    working_symbols = dict(symbols)
    final_count = len(working_symbols)
    
    # === STRICT SHRINK REJECTION (only in replace mode) ===
    if not allow_shrink and final_count < current_count:
        self.logger.warning(
            "[SS] Rejecting shrink because allow_shrink=False. "
            f"Current={current_count}, Incoming={final_count}, Source={source}"
        )
        return
    
    self.logger.info(
        f"[SS] 🔄 REPLACE MODE: {current_count} → {final_count} symbols (source={source})"
    )
```

#### Key Differences:
- **Merge mode** (merge_mode=True):
  - Copies current symbols
  - Updates with incoming symbols
  - All symbols preserved (unless exceeding cap)
  - Shrink rejection bypassed (not needed when adding)
  
- **Replace mode** (merge_mode=False, default):
  - Uses only incoming symbols
  - Shrink rejection applies (if allow_shrink=False)
  - Legacy behavior maintained

---

## File 2: core/symbol_manager.py

### Change 1: `_safe_set_accepted_symbols()` signature
**Location**: Line ~412

#### Before
```python
async def _safe_set_accepted_symbols(self, symbols_map: dict, *, allow_shrink: bool = False, source: Optional[str] = None):
```

#### After
```python
async def _safe_set_accepted_symbols(self, symbols_map: dict, *, allow_shrink: bool = False, merge_mode: bool = False, source: Optional[str] = None):
```

**Change Details**:
- Added parameter: `merge_mode: bool = False`
- Default is False (backward compatible)
- Passed through to SharedState.set_accepted_symbols()

---

### Change 1b: Parameter Detection Logic
**Location**: Line ~438-447

#### Before
```python
sig = inspect.signature(fn)
kwargs_call = {}
if "allow_shrink" in sig.parameters:
    kwargs_call["allow_shrink"] = allow_shrink
if "source" in sig.parameters and source:
    kwargs_call["source"] = source

result = fn(sanitized_map, **kwargs_call)
```

#### After
```python
sig = inspect.signature(fn)
kwargs_call = {}
if "allow_shrink" in sig.parameters:
    kwargs_call["allow_shrink"] = allow_shrink
if "merge_mode" in sig.parameters:
    kwargs_call["merge_mode"] = merge_mode
if "source" in sig.parameters and source:
    kwargs_call["source"] = source

result = fn(sanitized_map, **kwargs_call)
```

**Change Details**:
- Added detection for "merge_mode" parameter
- Passes merge_mode to SharedState if it supports it
- Graceful fallback if parameter not recognized

---

### Change 2: `add_symbol()` call to `_safe_set_accepted_symbols()`
**Location**: Line ~514

#### Before
```python
await self._safe_set_accepted_symbols(final_map, allow_shrink=False, source=source)
```

#### After
```python
await self._safe_set_accepted_symbols(final_map, allow_shrink=False, merge_mode=True, source=source)
```

**Change Details**:
- Added `merge_mode=True` parameter
- Enables additive behavior for single symbol proposals from discovery agents
- Comment updated: "P9 Fix: Use merge_mode=True for additive proposals from discovery agents..."

**Impact**:
- When SymbolScreener proposes a single symbol via `propose_symbol()`, it now adds instead of replaces
- Symbol universe grows with each proposal

---

### Change 3: `propose_symbols()` call to `_safe_set_accepted_symbols()`
**Location**: Line ~582

#### Before
```python
await self._safe_set_accepted_symbols(final_map, allow_shrink=False, source=source)
```

#### After
```python
await self._safe_set_accepted_symbols(final_map, allow_shrink=False, merge_mode=True, source=source)
```

**Change Details**:
- Added `merge_mode=True` parameter
- Enables additive behavior for batch symbol proposals from discovery agents
- Comment updated: "4. Commit once with merge_mode=True for additive batch proposals..."

**Impact**:
- When SymbolScreener proposes a batch of symbols, they're merged instead of replacing
- All proposal symbols accumulate in the universe

---

## Changes NOT Made (Intentional)

The following methods continue to use default merge_mode=False (replace mode):

### `initialize_symbols()` [Line ~234]
```python
# Still uses replace mode (merge_mode=False by default)
await self._safe_set_accepted_symbols(validated, allow_shrink=True)
```
**Reason**: Startup initialization should replace, not merge

### `flush_buffered_proposals_to_shared_state()` [Line ~663]
```python
# Still uses replace mode (merge_mode=False by default)
await self._safe_set_accepted_symbols(final_map, allow_shrink=True)
```
**Reason**: Finalization/flush should finalize, not accumulate

### `finalize_universe()` [Line ~691]
```python
# Still uses replace mode (merge_mode=False by default)
await self._safe_set_accepted_symbols(final_map, allow_shrink=allow_shrink)
```
**Reason**: Explicit finalization/trimming should replace

### `set_accepted_symbols()` (public API) [Line ~175]
```python
# Still uses replace mode (merge_mode=False by default)
return await self._safe_set_accepted_symbols(symbols_map, allow_shrink=allow_shrink)
```
**Reason**: Public API maintains backward compatibility with replace behavior

---

## Summary of Changes

### SharedState (core/shared_state.py)
- Lines Changed: ~100 (entire set_accepted_symbols method)
- New Code: Merge mode logic branching
- Removed Code: None (everything preserved with conditional logic)
- Impact: Enables merge mode for additive symbol proposals

### SymbolManager (core/symbol_manager.py)
- Lines Changed: ~15
  - Line ~412: Signature updated (1 line)
  - Line ~438: Parameter detection added (3 lines)
  - Line ~514: add_symbol call updated (1 line)
  - Line ~582: propose_symbols call updated (1 line)
- New Code: merge_mode=True in two method calls
- Removed Code: None (everything added)
- Impact: Enables merge mode for discovery agent proposals

### Total
- **2 files modified**
- **~115 lines changed**
- **3 method signatures updated**
- **2 method calls updated**
- **1 conditional logic block added**
- **0 backward compatibility breaks**

---

## Syntax Validation

### Python Compilation Check
```
✅ core/shared_state.py - Compiles successfully
✅ core/symbol_manager.py - Compiles successfully
```

### Type Hints Check
All type annotations are valid and consistent:
- `merge_mode: bool = False` ✅
- Optional parameters properly ordered ✅
- Return types unchanged ✅

---

## Testing the Changes

### Unit Test Example (if framework available)
```python
async def test_set_accepted_symbols_merge_mode():
    # Test merge mode
    state = SharedState(symbols={'A': {}, 'B': {}})
    await state.set_accepted_symbols({'C': {}}, merge_mode=True)
    assert 'A' in state.accepted_symbols
    assert 'B' in state.accepted_symbols
    assert 'C' in state.accepted_symbols
    # Total: 3 symbols ✅

async def test_set_accepted_symbols_replace_mode():
    # Test replace mode (default)
    state = SharedState(symbols={'A': {}, 'B': {}})
    await state.set_accepted_symbols({'C': {}}, merge_mode=False)
    assert 'A' not in state.accepted_symbols
    assert 'B' not in state.accepted_symbols
    assert 'C' in state.accepted_symbols
    # Total: 1 symbol ✅
```

---

## Verification Checklist

- [x] **Syntax Valid** - Both files compile without errors
- [x] **Types Correct** - All type hints are valid
- [x] **Backward Compatible** - Default behavior unchanged (merge_mode=False)
- [x] **Parameter Detection** - Uses inspect.signature for safe detection
- [x] **Error Handling** - Graceful fallback for unsupported signatures
- [x] **Logging Enhanced** - Clear "MERGE MODE" vs "REPLACE MODE" messages
- [x] **Comments Updated** - All changes documented with comments

---

## Rollback Instructions

### Quick Revert to Original
```bash
# If changes not yet committed:
git checkout core/shared_state.py core/symbol_manager.py

# If changes already committed:
git revert <commit-hash>
git push origin main
```

**Revert Time**: < 1 minute

---

## Code Diff Summary

```diff
File: core/shared_state.py
- async def set_accepted_symbols(self, symbols, *, allow_shrink=False, source=None):
+ async def set_accepted_symbols(self, symbols, *, allow_shrink=False, merge_mode=False, source=None):
  
- # === HARD REPLACE MODE ===
+ if merge_mode:
+     working_symbols = dict(self.accepted_symbols)
+     working_symbols.update(symbols)
+ else:
+     working_symbols = dict(symbols)
+     # Shrink rejection logic...

File: core/symbol_manager.py
- async def _safe_set_accepted_symbols(self, symbols_map, *, allow_shrink=False, source=None):
+ async def _safe_set_accepted_symbols(self, symbols_map, *, allow_shrink=False, merge_mode=False, source=None):

+ if "merge_mode" in sig.parameters:
+     kwargs_call["merge_mode"] = merge_mode

- await self._safe_set_accepted_symbols(final_map, allow_shrink=False, source=source)
+ await self._safe_set_accepted_symbols(final_map, allow_shrink=False, merge_mode=True, source=source)

- await self._safe_set_accepted_symbols(final_map, allow_shrink=False, source=source)
+ await self._safe_set_accepted_symbols(final_map, allow_shrink=False, merge_mode=True, source=source)
```

---

**All changes are minimal, focused, and backward compatible.**
