# Bottleneck Fixes - Phase 2: Recovery/Rotation Unblocking

**Date:** April 24, 2026  
**Target:** Unblock forced recovery exits, align rotation policies, fix entry-sizing config

---

## Issue Summary

Three critical bottlenecks prevent clean rotation and recovery:

1. **Min-hold Pre-execution Gates** → Block recovery/rotation SELL intents
   - `meta_controller.py:12525` (liquidity restoration)
   - `meta_controller.py:11862` (stagnation exit batching)
   - **Impact:** Capital cannot be recycled when needed for rotation

2. **Strict One-Position Gate** → Frequent `POSITION_ALREADY_OPEN` rejections
   - `meta_controller.py:13856`
   - **Impact:** New entries can't execute even when rotation intends to free slots

3. **Micro Bracket Rotation Restriction** → Blocks rotation unless forced flag set
   - `rotation_authority.py:176`
   - **Impact:** Forced exits can still be gated despite having `_force_micro_rotation` flag

4. **Entry-Sizing Config Misalignment** → Floor is misaligned with runtime normalization
   - `config.py:1360` (MIN_ENTRY_USDT floor check)
   - `.env:45, 48, 50, 140` (low-size defaults)
   - **Impact:** Config intent unclear; runtime corrects but code is brittle

---

## Fix Sequence

### Fix #1: Safe Min-Hold Bypass for Forced Recovery Exits

**File:** `core/meta_controller.py`  
**Sections:** Lines 12520-12530, 11857-11870

**Change:**
- Add `_bypass_min_hold` flag to recovery signals  
- Make `_safe_passes_min_hold()` respect the bypass flag  
- Apply to LIQUIDITY_RESTORE and STAGNATION forced exits

**Reasoning:**
- Forced recovery exits must not be blocked by min-hold—they *are* the recovery mechanism
- This bypasses pre-decision gate but still respects execution-path safety

---

### Fix #2: Micro Rotation Override Policy Wiring

**File:** `core/rotation_authority.py`  
**Section:** Lines 171-186, 313-340

**Change:**
- Strengthen `force_rotation` override logic to always allow if flag set
- Add explicit logging when override occurs
- Wire up micro bracket restriction to honor `_force_micro_rotation` intent

**Reasoning:**
- When micro rotations are forced (e.g., capacity escape), they must succeed
- Current logic allows override but isn't clear about precedence

---

### Fix #3: Entry-Sizing Config & Profile Alignment

**File:** `core/config.py` (lines 1360) + `.env` (lines 45, 48, 50, 140)

**Change:**
- Raise DEFAULT_PLANNED_QUOTE from 12 → 25  
- Raise MIN_ENTRY_USDT from 10 → 25  
- Raise MIN_ENTRY_QUOTE_USDT from 10 → 25  
- Align EMIT_BUY_QUOTE with floor  
- Add comment explaining floor alignment

**Reasoning:**
- Config defaults should match runtime floor expectations
- Reduces runtime normalization churn and makes intent clearer
- Aligns with SIGNIFICANT_POSITION_FLOOR = 25 USDT

---

## Validation Checklist

After applying fixes:

- [ ] `python3 -m compileall -q core agents utils` → cleanly compiles
- [ ] Orchestrator module imports successfully
- [ ] No new syntax or import errors
- [ ] Recovery/forced exits now carry `_bypass_min_hold` flag
- [ ] Rotation override logs show correct precedence
- [ ] `.env` config aligns with runtime floor expectations

---

## Risk Assessment

**Low:** These are guard logic improvements + config alignment
- No algorithm changes
- Respects existing decision gates
- Forces are already authorized upstream
- Improves clarity without changing flow

---
