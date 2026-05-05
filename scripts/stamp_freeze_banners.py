#!/usr/bin/env python3
"""Stamp a freeze-status banner into every LEGACY / QUARANTINED file.

Idempotent — re-runnable. Reads MODULE_FREEZE_MANIFEST.json and inserts a
single comment block near the top of each non-ACTIVE/non-WRAPPED file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MODULE_FREEZE_MANIFEST.json"
MARKER = "# === OCTIVAULT FREEZE BANNER ==="

BANNER_TEMPLATE = """{marker}
# STATUS:    {status}
# CANONICAL: {canonical}
# REASON:    {reason}
# POLICY:    See STEP_4_MODULE_FREEZE.md — do not import from main.py / top-level scripts.
# {marker_end}
"""


def banner_for(status: str, canonical: str, reason: str) -> str:
    return BANNER_TEMPLATE.format(
        marker=MARKER,
        marker_end="=" * len(MARKER[2:]),
        status=status,
        canonical=canonical,
        reason=reason,
    )


def stamp_python(path: Path, banner: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False  # already stamped
    lines = text.splitlines(keepends=True)
    insert_at = 0
    # Skip shebang
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    # Skip encoding line
    if insert_at < len(lines) and "coding" in lines[insert_at] and lines[insert_at].startswith("#"):
        insert_at += 1
    # Skip module docstring
    if insert_at < len(lines) and lines[insert_at].lstrip().startswith(('"""', "'''")):
        quote = lines[insert_at].lstrip()[:3]
        # single-line docstring
        if lines[insert_at].count(quote) >= 2:
            insert_at += 1
        else:
            insert_at += 1
            while insert_at < len(lines) and quote not in lines[insert_at]:
                insert_at += 1
            insert_at += 1  # past closing """
    new_lines = lines[:insert_at] + [banner + "\n"] + lines[insert_at:]
    path.write_text("".join(new_lines), encoding="utf-8")
    return True


def stamp_shell(path: Path, banner: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    lines = text.splitlines(keepends=True)
    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    new_lines = lines[:insert_at] + [banner + "\n"] + lines[insert_at:]
    path.write_text("".join(new_lines), encoding="utf-8")
    return True


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    stamped = skipped = missing = 0
    for family in manifest["families"].values():
        canonical = family["canonical"]
        for mod in family["modules"]:
            status = mod["status"]
            if status in ("ACTIVE", "WRAPPED"):
                continue
            path = ROOT / mod["path"]
            if not path.exists():
                missing += 1
                print(f"  ⚠️  missing: {mod['path']}")
                continue
            banner = banner_for(status, canonical, mod["reason"])
            if path.suffix == ".py":
                changed = stamp_python(path, banner)
            elif path.suffix in (".sh", ".command"):
                changed = stamp_shell(path, banner)
            else:
                print(f"  ?  unknown ext: {mod['path']}")
                continue
            if changed:
                stamped += 1
                print(f"  ✓ {status:11s}  {mod['path']}")
            else:
                skipped += 1
    print(f"\nstamped: {stamped}   already-stamped: {skipped}   missing: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
