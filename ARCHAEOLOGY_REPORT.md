# 🔬 Codebase Archaeology — Executive Summary
_Generated 2026-05-05_

## TL;DR

| Metric | Value |
|---|---:|
| Real Python files (excl. `.claude/` worktrees) | **316** |
| Files reachable from live entry point | **145 (45%)** |
| Files NOT reachable from entry point | **171 (54%)** |
| Files with `__main__` block (mostly diagnostic one-shots) | 81 |
| Markdown files at root (most are historical) | ~400 |
| `.claude/worktrees/` noise that bloated counts | 122 MB / 496 .py |

> **Real entry point:** `🎯_MASTER_SYSTEM_ORCHESTRATOR.py`
> All three launcher scripts (`START_TRADING.sh`, `launch_growth_mode.sh`, `run_bot_resilient.sh`) call only this file. Nothing else is in production.

> 🚨 **Smoking-gun contradiction confirmed:** `master_orchestrator.py` and `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` are **byte-identical** (MD5 `eeefb89712ea1f46f2d2ead96fa7cebf`). The plain-named one is a stale clone. Anyone editing it changes nothing — exactly the "contradictions / scripts that don't work" feeling you described. **Recommend quarantining `master_orchestrator.py` immediately.**

---

## What we know now (Phase 1 complete)

✅ Tagged baseline: `baseline-archaeology-20260505`
✅ Identified the single real entry point
✅ Built static dependency closure (`_archaeology/live_dependency_closure.txt`)
✅ Listed all 171 unreachable files (`_archaeology/unreached_from_entry.txt`)
✅ Detected 19 patch-artifacts by filename, 61 orphans, 3 duplicate name groups
✅ Generated 23 obvious safe-to-quarantine candidates at root level

---

## The codebase has 4 distinct populations

### 1. 🟢 Live core (~145 files) — the working system
Mostly under `src/l0_core` … `src/l8_lifecycle`. **This is your actual product.** It already has the clean layered architecture (l0=core, l1=exchange, l2=marketdata, l3=portfolio, l4=execution, l5=strategy, l6=governance, l7=observability, l8=lifecycle). Good news: the bones are healthy.

### 2. 🟠 Junk at root (23 files) — quarantine NOW
`FIX_*`, `verify_*`, `diagnose_*`, `phase1_*`, `force_*`, `_test_*`, `restore_*`, `launch_with_*`, `CAPITAL_ALLOCATOR_FIX_CODE.py`, etc. None are imported, none are in the live closure. **Pure historical patch debris.** See `_archaeology/quarantine_plan.txt`.

### 3. 🟡 One-shot tools / monitors in `src/l7/l8` (~10 files)
`monitor_4hour_session.py`, `verify_dust_fix.py`, `apply_recovery_to_live.py`, etc. Probably useful but not part of the live trading loop. Should be renamed/relocated to `tools/` with clear "ad-hoc" labelling rather than living next to live code.

### 4. 🔴 Documentation chaos (~400 .md files at root)
`6HOUR_FINAL_REPORT.md`, `CAPITAL_DECAY_DIAGNOSIS.md`, `ANALYSIS_SUMMARY_READY_TO_FIX.md`, etc. — patch-by-patch incident notes. Should be moved en masse to `_archive/docs/` leaving only `README.md`, `RUNBOOK.md`, `ARCHITECTURE_REALITY.md`.

---

## Recommended next steps (in order)

### Step A — Apply the safe quarantine (1 command, fully reversible)
```bash
python3 _archaeology/quarantine.py --apply
git commit -m "chore: quarantine 23 unreachable patch-artifacts at root (Phase 1)"
# Run the bot for 24h. If anything breaks, just `git revert`.
```

### Step B — Mass-archive the docs
```bash
mkdir -p _archive/docs
git mv [0-9]*HOUR_*.md ANALYSIS_*.md CAPITAL_*.md AUTO_*.md DUST_*.md \
       FIX_*.md FOURTH_*.md *_FIX_*.md *_DIAGNOSIS.md *_DEPLOYMENT*.md \
       *_REPORT.md *_SUMMARY*.md _archive/docs/ 2>/dev/null
git commit -m "docs: archive ~300 historical incident reports"
```

### Step C — Decide on the 32 unreachable files in `src/`
Most are `__init__.py` (false positives — keep) and one-shot monitor/verify scripts. Move the latter to `tools/` and prefix with `oneshot_`:
```bash
git mv src/l7_observability/monitors/monitor_4hour_session.py tools/oneshot_monitor_4hour_session.py
git mv src/l8_lifecycle/runners/verify_*.py tools/
```

### Step D — Lock in current behavior with characterization tests
Before any refactor, snapshot what the live system does today. I can scaffold this with `pytest` + recorded fixtures.

### Step E — Set up guardrails so the rot doesn't return
- `pre-commit` with `ruff` (kills unused imports automatically)
- `vulture` in CI (flags dead code in PRs)
- Convention: **all one-shot scripts go in `tools/` with `oneshot_` prefix**, never at repo root.

---

## Files to read for full detail

| File | Purpose |
|---|---|
| `ARCHITECTURE_REALITY.md` | Full inventory with status per file |
| `_archaeology/live_dependency_closure.txt` | The 145 live files |
| `_archaeology/unreached_from_entry.txt` | The 171 candidates for review |
| `_archaeology/quarantine_plan.txt` | The 23 safe-to-move files |
| `_archaeology/patch_artifacts.txt` | 19 files screaming "FIX/BACKUP/etc." |

---

## Decision needed from you

1. **Apply Step A now?** (quarantine 23 obvious junk files at root) — _safe & reversible_
2. **Apply Step B?** (mass-archive ~300 historical .md files) — _safe & reversible_
3. **Should I scaffold characterization tests (Step D)?**
4. **Should I set up pre-commit + ruff guardrails (Step E)?**

Reply with the step letters you want me to execute (e.g. "A B E").
