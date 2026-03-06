# 📚 SHADOW MODE DUPLICATE EMISSION FIX - DOCUMENTATION INDEX

**Generated:** March 3, 2026  
**Status:** ⏳ AWAITING CODE MERGE  
**Severity:** 🔴 CRITICAL

---

## 🚀 Quick Links

### For The Busy (5 minutes)
👉 **Start here:** [`SHADOW_MODE_FIX_QUICK_ACTION.md`](./SHADOW_MODE_FIX_QUICK_ACTION.md)
- One-sentence problem
- One-sentence fix
- 18 lines to delete
- Done in 5 minutes

### For Implementers (30 minutes)
👉 **Read here:** [`SHADOW_MODE_DUPLICATE_EMISSION_FIX_PATCH.md`](./SHADOW_MODE_DUPLICATE_EMISSION_FIX_PATCH.md)
- Full technical guide
- Code examples (before/after)
- Verification procedures
- Deployment checklist

### For Project Managers (10 minutes)
👉 **Check here:** [`SHADOW_MODE_DUPLICATE_EMISSION_STATUS.md`](./SHADOW_MODE_DUPLICATE_EMISSION_STATUS.md)
- Current status
- Task breakdown
- Timeline
- Handoff checklist

### For Visual Learners (15 minutes)
👉 **See here:** [`SHADOW_MODE_VISUAL_GUIDE.md`](./SHADOW_MODE_VISUAL_GUIDE.md)
- Flow diagrams (before/after)
- Code comparison
- Impact visualization
- Decision tree

---

## 📋 File Descriptions

| File | Audience | Read Time | Purpose |
|------|----------|-----------|---------|
| `SHADOW_MODE_FIX_QUICK_ACTION.md` | Everyone | 5 min | Quick reference card |
| `SHADOW_MODE_DUPLICATE_EMISSION_FIX_PATCH.md` | Developers | 30 min | Full implementation guide |
| `SHADOW_MODE_DUPLICATE_EMISSION_STATUS.md` | PMs/Leads | 10 min | Project status & timeline |
| `SHADOW_MODE_VISUAL_GUIDE.md` | Engineers/Architects | 15 min | Diagrams and visualizations |
| `00_SHADOW_MODE_DUPLICATE_EMISSION_FIX_AWAITING_MERGE.md` | Stakeholders | 20 min | Comprehensive overview |

---

## 🎯 The Problem in 30 Seconds

**Shadow mode calls `_emit_trade_executed_event()` twice:**

1. Once explicitly in `_place_with_client_id()` (WRONG)
2. Once implicitly in `_handle_post_fill()` (CORRECT)

**Result:** All accounting runs twice → NAV explodes 5x → System broken

**Fix:** Delete the first emission (18 lines) → Keep the second → Done

---

## ✅ The Fix in 30 Seconds

**Delete this block from `_place_with_client_id()` shadow mode section:**

```python
try:
    await self._emit_trade_executed_event(
        symbol=symbol,
        side=side,
        tag=tag,
        order=simulated,
    )
    self.logger.info(f"[EM:ShadowMode:Canonical]...")
except Exception as e:
    self.logger.error(f"[EM:ShadowMode:EmitFail]...")
    if bool(self._cfg("STRICT_OBSERVABILITY_EVENTS", False)):
        raise
```

**Keep the `_handle_post_fill()` block** → It emits internally → Done

---

## 📊 Documentation Map

```
SHADOW MODE DUPLICATE EMISSION FIX
│
├─ QUICK ACTION (5 min)
│  └─ SHADOW_MODE_FIX_QUICK_ACTION.md
│
├─ IMPLEMENTATION (30 min)
│  └─ SHADOW_MODE_DUPLICATE_EMISSION_FIX_PATCH.md
│
├─ PROJECT STATUS (10 min)
│  └─ SHADOW_MODE_DUPLICATE_EMISSION_STATUS.md
│
├─ VISUAL GUIDE (15 min)
│  └─ SHADOW_MODE_VISUAL_GUIDE.md
│
└─ OVERVIEW (20 min)
   └─ 00_SHADOW_MODE_DUPLICATE_EMISSION_FIX_AWAITING_MERGE.md
```

---

## 🎬 Step-by-Step Workflow

### Step 1: Wait for Code Merge
```bash
grep -n "[EM:ShadowMode:Canonical]" core/execution_manager.py
```

### Step 2: Apply Fix (5 minutes)
1. Search: `[EM:ShadowMode:Canonical]`
2. Find: try: block ABOVE this log line
3. Delete: Entire try-except (18 lines)
4. Keep: `_handle_post_fill()` try-except

### Step 3: Run Tests (5 minutes)
```bash
pytest tests/test_shadow_mode.py -v
```

### Step 4: Verify NAV (5 minutes)
Expected: 107 → ~105.99 (not 557)

### Step 5: Deploy
Staging → Production

---

## ❓ Quick FAQ

| Q | A |
|---|---|
| What's the bug? | `_emit_trade_executed_event()` called twice (2x accounting) |
| What's the fix? | Delete the first emission, keep the second |
| How many lines? | 18 lines to delete |
| Will it break? | No - emission still happens via `_handle_post_fill()` |
| How long? | 5 min to apply, 30 min to test, 1 hour to deploy |
| Is it urgent? | Yes - blocks shadow mode usage |

---

**Status:** Ready to apply  
**Next:** Code merge → Apply fix → Test → Deploy  
**Timeline:** ~2 hours once code is merged
