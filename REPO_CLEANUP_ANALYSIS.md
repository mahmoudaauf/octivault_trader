# Repository Cleanup Analysis

**Date:** April 24, 2026  
**Status:** Repo feels "dirty" due to multiple issues

---

## Why the Repository Feels Dirty

### 1. **Uncommitted Changes**
```
Changes to be staged:  2 files (test deletions)
Changes not staged:    100+ files (deletions, modifications)
Untracked files:       ~200 generated model files (.pkl)
```

### 2. **Unpushed Commits**
```
Your branch is ahead of 'origin/main' by 16 commits
```

### 3. **Inconsistent State**
```
- Files deleted from disk but tracked by git
- Modified files not staged
- Generated model files cluttering the repo
- Old documentation/guides taking up space
```

---

## Detailed Breakdown

### 📋 Staged for Deletion (2 files)
These are ready to commit but not yet pushed:
```
- tests/test_regime_system.py
- tests/test_week3_integration_templates.py
```

### 🗑️ Unstaged Deletions (~50 files)
Files deleted from working directory but git still tracks:

**Documentation (guides & reports):**
- ISSUE_19_*.md (4 files)
- ISSUE_20_*.md (1 file)
- ISSUE_21_*.md (1 file)
- PHASE_2D_*.md (6 files)
- SPRINT_1_*.md (2 files)

**Agents (14 deleted):**
- arbitrage_hunter_agent.py
- check_symbol_usage.py
- cot_assistant.py
- ipo_forecaster.py
- news_reactor.py
- refactor_symbol_feed.py
- rl_strategist.py
- signal_fusion_agent.py
- signal_utils.py
- symbol_discoverer_agent.py
- (+ others)

**Core modules (13+ deleted):**
- Files in core/_archived/
- ab_tester.py
- Various backup/obsolete files

**Root-level files:**
- diagnostic_signal_flow.py
- error_monitor.py
- phase2_monitoring.py
- phase2_paper_trading.py
- phase3_live_trading.py
- test_*.py scripts

### ⚙️ Modified Files (Not Staged)
```
- core/agent_manager.py
- core/agent_registry.py
- core/config.py ← (just modified for Phase 2 fixes!)
- core/execution_authority.py
- core/meta_controller.py ← (just modified for Phase 2 fixes!)
- core/rotation_authority.py ← (just modified for Phase 2 fixes!)
- agents/dip_sniper.py
- agents/ml_forecaster.py
- agents/swing_trade_hunter.py
- agents/trend_hunter.py
- agents/wallet_scanner_agent.py
+ more...
```

### 📦 Generated Model Files (~200 files, ~500 MB+)
These should be in `.gitignore`:
```
models/mlforecaster_*.pkl         (~35 files)
models/trendhunter_*.pkl          (~65+ files)
models/_incompatible_quarantine/  (directory)
```

---

## What's Making It Feel "Dirty"

### ✗ Problem 1: Too Many Deletions
Over 50 files marked as deleted in git — suggests aggressive refactoring or cleanup that wasn't properly staged/committed.

### ✗ Problem 2: Mixed States
Some files deleted, some modified, some staged — unclear commit history ahead of main.

### ✗ Problem 3: Generated Files in Repo
ML model metadata files (.pkl) shouldn't be version-controlled. These are:
- Large (hundreds of MB)
- Environment-specific
- Regenerated on each run
- Polluting the working directory

### ✗ Problem 4: Orphaned Documentation
Old guides/reports for phases that are complete but not cleaned up:
- ISSUE_*.md files
- PHASE_2D_*.md files
- SPRINT_*.md files

### ✗ Problem 5: Unpushed Commits
16 commits ahead of origin/main = local changes not synced to remote.

---

## Cleanup Recommendations

### 🎯 Quick Cleanup (5 minutes)

#### 1. Clean up generated model files
```bash
# Remove all .pkl files (they regenerate automatically)
find models -name "*.pkl" -delete
# Or if you want to keep them locally:
git rm --cached models/*.pkl
echo "models/*.pkl" >> .gitignore
```

#### 2. Commit the staged deletions
```bash
git commit -m "Clean: Remove obsolete test files and archived code"
```

#### 3. Clean up old documentation
```bash
git rm ISSUE_*.md PHASE_2D_*.md SPRINT_*.md
git commit -m "Clean: Remove old phase documentation"
```

### 🧹 Thorough Cleanup (15 minutes)

```bash
# 1. Stage all deletions
git add -A

# 2. Clean up .gitignore to exclude generated files
cat >> .gitignore << 'EOF'
# Generated files - regenerate per environment
models/*.pkl
models/_incompatible_quarantine/

# Runtime/debug files
phase2_*.py
phase3_*.py
diagnostic_*.py
error_monitor.py

# Test artifacts
test_result.txt
.pytest_cache/
EOF

# 3. Remove tracked but ignored files
git rm --cached models/*.pkl 2>/dev/null || true

# 4. Commit cleanup
git add .gitignore
git commit -m "Clean: Remove generated/environment-specific files from tracking"

# 5. Push to remote
git push origin main
```

### 📋 Strategic Cleanup (Full)

```bash
# Create a clean branch for major cleanup
git checkout -b cleanup/remove-generated-files

# 1. Remove all generated model files from tracking
git rm --cached -r models/*.pkl 2>/dev/null || true

# 2. Remove old phase documentation
git rm ISSUE_*.md PHASE_2D_*.md SPRINT_*.md SPRINT_*.py 2>/dev/null || true

# 3. Remove debug/obsolete scripts
git rm diagnostic_signal_flow.py error_monitor.py 2>/dev/null || true
git rm phase2_*.py phase3_*.py 2>/dev/null || true
git rm test_*.py 2>/dev/null || true

# 4. Update .gitignore
cat >> .gitignore << 'EOF'
# ML Model metadata (regenerated per environment)
models/mlforecaster_*.pkl
models/trendhunter_*.pkl
models/_incompatible_quarantine/
models/*.pkl

# Runtime monitoring/debug scripts
diagnostic_*.py
error_monitor.py
phase2_*.py
phase3_*.py
test_trading.sh
run-local.sh

# Test artifacts
test_result.txt
.pytest_cache/
*.pyc
__pycache__/

# Environment files
.env.local
.env.*.local
EOF

# 5. Commit all changes
git add .
git commit -m "Clean: Remove generated files and archive old documentation"

# 6. Push and create PR
git push origin cleanup/remove-generated-files
```

---

## Before & After

### Before Cleanup
```
On branch main
Your branch is ahead of 'origin/main' by 16 commits

Changes to be committed: 2 files
Changes not staged:     100+ files
Untracked files:        200+ .pkl files
```

### After Cleanup
```
On branch main
Your branch is ahead of 'origin/main' by 2 commits (clean history)

Changes to be committed: 0 files
Changes not staged:     0 files
Untracked files:        (only legitimate untracked, no generated files)
```

---

## Why This Happened

1. **ML Training**: Model files generated during ML experiments (.pkl files)
2. **Aggressive Refactoring**: Many agents/utilities removed without cleanup commit
3. **Phase Documentation**: Accumulated docs from multiple phases not archived
4. **No .gitignore**: Generated files weren't excluded from tracking
5. **Unpushed History**: 16 commits waiting to be pushed

---

## Recommended Next Steps

### ✅ Immediate (Do This Now)
```bash
# 1. Check what you REALLY need to keep
git status | grep "deleted:" | wc -l  # See how many deletions

# 2. Create a cleanup branch
git checkout -b cleanup/repo-maintenance

# 3. Add proper .gitignore
# (Update .gitignore as shown above)

# 4. Remove generated files
git rm --cached models/*.pkl 2>/dev/null || true
git add .gitignore
git commit -m "Clean: Exclude generated ML model files from tracking"

# 5. Push and merge
git push origin cleanup/repo-maintenance
# Create PR, review, merge
```

### 📋 Ongoing
- Always add generated/cache files to `.gitignore` before committing
- Archive old documentation to `docs/archived/` or separate branch
- Keep `.gitignore` updated with environment-specific files
- Regular cleanup commits to keep history clean

---

## Summary

Your repo feels dirty because:

| Issue | Count | Size | Fix |
|-------|-------|------|-----|
| Deleted files tracked | 50+ | - | Stage & commit |
| Modified files unstaged | 10+ | - | Stage & commit |
| Generated .pkl files | 200+ | ~500MB | Add to .gitignore |
| Old documentation | 15+ | ~1MB | Archive or delete |
| Unpushed commits | 16 | - | Push to origin |

**Recommended Action:** Run the "Thorough Cleanup" script above (15 minutes)

This will make your repo clean, organized, and ready for future work.

---

**Generated:** April 24, 2026  
**Status:** Ready for cleanup recommendation
