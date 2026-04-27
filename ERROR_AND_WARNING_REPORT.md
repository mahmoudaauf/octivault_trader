# Error and Warning Report
**Generated:** April 19, 2026

## Summary
- **Total Python Files:** 14,864
- **Files with Syntax Errors:** 4 (in project; 7 in venv)
- **Critical Issues:** 4

## Critical Syntax Errors in Project

### 1. ❌ `core/_archived/symbol_filter_pipeline.py` (Line 1)
**Error:** `IndentationError: unexpected indent`
- **Status:** ARCHIVED FILE
- **Action:** Review file structure - appears to have incorrect leading whitespace

### 2. ❌ `core/_archived/execution_manager_backup.py` (Line 85)
**Error:** `IndentationError: unindent does not match any outer indentation level`
- **Status:** ARCHIVED FILE
- **Location:** Line 85 - incorrect indentation in variable assignment
```python
step = float(filters.step_size or 0.0) # BOOTSTRAP FIX: Handle 0.0 step
price_safe = price if price > 0 else 1.0 # Guard zero div
    estimated_qty = spend / price_safe  # ❌ EXTRA INDENT
```
- **Issue:** Mixed indentation levels - lines 86-87 have incorrect indentation
- **Lines with Issues:** 87, 90-91, 93, 97-98

### 3. ❌ `core/_archived/execution_manager_dedented.py` (Line 176)
**Error:** `IndentationError: unexpected unindent`
- **Status:** ARCHIVED FILE
- **Location:** Around line 176 in exception handler
```python
except Exception:
pass  # ❌ MISSING INDENT

trade_recorded = False
```
- **Issue:** Statement after exception block has inconsistent indentation

### 4. ❌ `core/_archived/meta_controller_fixed.py` (Line 976)
**Error:** `IndentationError: expected an indented block`
- **Status:** ARCHIVED FILE
- **Location:** Line 976 - missing indentation for method docstring
```python
def _count_significant_positions(self) -> Tuple[int, int, int]:
"""  # ❌ SHOULD BE INDENTED
```

## Venv Errors (Ignorable)
- 7 syntax errors in `venv/lib/python3.9/site-packages/ccxt/...` 
- These are from external dependencies and don't affect your code
- Recommendation: These can be ignored as they're in third-party libraries

## Recommendations

1. **Archive Files:** All 4 errors are in `_archived/` directory - these are backup/old files
2. **Action Items:**
   - Option A: Delete archived backup files if no longer needed
   - Option B: Fix indentation in archived files for completeness
   
3. **Active Code:** No syntax errors found in active source code

4. **Next Steps:**
   - Run type checking with mypy for warnings
   - Check for unused imports and dead code
   - Review logging for runtime warnings

