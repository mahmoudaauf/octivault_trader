"""Convention enforcement for OctiVault codebase.

All ad-hoc/diagnostic/one-shot Python scripts MUST be named with oneshot_ prefix
and placed in tools/ directory, NOT at the root or mixed in with core code.

Examples of one-shot scripts (candidates for tools/oneshot_*):
  - monitor_*.py (run once to check status)
  - verify_*.py (verification/audit scripts)
  - diagnose_*.py (troubleshooting scripts)
  - check_*.py (one-time checks)
  - force_*.py (emergency/recovery operations)
  - validate_*.py (one-time validation)

This hook prevents the creep of ad-hoc scripts into the repo root that makes
cleanup archaeology necessary later.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Filename patterns that indicate one-shot scripts
ONE_SHOT_PATTERNS = {
    "monitor_",
    "verify_",
    "diagnose_",
    "check_",
    "force_",
    "validate_",
    "apply_",
    "phase1_",
    "phase2_",
    "phase3_",
    "phase4_",
    "restore_",
    "launch_with_",
    "show_",
    "_test_",
}

# Directories where one-shot scripts are acceptable
ALLOWED_DIRS = {"tools", "tests", "_archaeology", "docs/archive/scripts"}


def check_file(filepath: str) -> bool:
    p = Path(filepath)
    name = p.name.lower()

    # Not a Python file
    if not name.endswith(".py"):
        return True

    # Check if it matches one-shot pattern
    is_one_shot = any(name.startswith(prefix) for prefix in ONE_SHOT_PATTERNS)

    if not is_one_shot:
        return True

    # It's a one-shot script. Check if it's in an allowed location.
    rel = p.relative_to(ROOT)
    first_part = rel.parts[0] if rel.parts else ""

    if first_part in ALLOWED_DIRS:
        return True

    # One-shot script at root or in disallowed location
    print(f"❌ Convention violation: {filepath}")
    print("   One-shot scripts must use 'oneshot_' prefix and live in tools/")
    print(f"   Current:  {filepath}")
    print(f"   Expected: tools/oneshot_{name}")
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: check-conventions.py <file> [<file> ...]")
        sys.exit(0)

    violations = []
    for filepath in sys.argv[1:]:
        if not check_file(filepath):
            violations.append(filepath)

    if violations:
        print(f"\n❌ {len(violations)} convention violation(s) found")
        sys.exit(1)
    else:
        print("✅ All files follow conventions")
        sys.exit(0)


if __name__ == "__main__":
    main()
