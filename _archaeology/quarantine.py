"""Quarantine SAFE candidates: files that are BOTH
  (a) named like a patch-artifact (FIX/OLD/BACKUP/verify_*/diagnose_*),
  AND (b) not in the live dependency closure.

Files are MOVED via `git mv` (history preserved) into
_archive/2026-05-05_archaeology/. Fully reversible with `git mv` back.

Excludes:
- __init__.py (package markers, not directly imported)
- anything under src/ (clean architecture - needs human review first)
- anything under tests/ (tests are by convention not "imported")
- anything under docs/, tools/ (build/dev infra)
"""
from __future__ import annotations
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "_archive" / "2026-05-05_archaeology"
ARCHIVE.mkdir(parents=True, exist_ok=True)

live = set((ROOT / "_archaeology" / "live_dependency_closure.txt").read_text().splitlines())
unreached = (ROOT / "_archaeology" / "unreached_from_entry.txt").read_text().splitlines()

EXCLUDE_PREFIXES = ("src/", "tests/", "docs/", "tools/", "agents/", "automation/")
EXCLUDE_NAMES = {"__init__.py"}

# Filename screams "junk patch artifact" - root level only
def is_obvious_junk(rel: str) -> bool:
    if "/" in rel:                           # only root-level
        return False
    if rel in EXCLUDE_NAMES:
        return False
    name = rel.upper()
    patterns = ("FIX_", "_FIX", "VERIFY_", "DIAGNOSE_", "FORCE_",
                "PHASE1_", "VALIDATE_", "_TEST_", "CHECK_", "RESTORE_",
                "LAUNCH_WITH_", "SHOW_DETECTED", "_BACKUP", "BACKUP_",
                "CAPITAL_ALLOCATOR_FIX")
    return any(p in name for p in patterns)

candidates = [r for r in unreached if is_obvious_junk(r)]

print(f"Quarantine candidates ({len(candidates)}):")
for c in candidates:
    print(f"  {c}")

(ROOT / "_archaeology" / "quarantine_plan.txt").write_text("\n".join(candidates))
print(f"\nPlan written to _archaeology/quarantine_plan.txt")
print("Run with --apply to actually move files.")

import sys
if "--apply" in sys.argv:
    moved = 0
    for c in candidates:
        src = ROOT / c
        dst = ARCHIVE / c
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(["git", "mv", str(src), str(dst)], cwd=ROOT, check=True)
            moved += 1
        except subprocess.CalledProcessError as e:
            print(f"  ⚠ skip {c}: {e}")
    print(f"\n✅ Moved {moved} files to {ARCHIVE.relative_to(ROOT)}")
