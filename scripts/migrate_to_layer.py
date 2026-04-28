"""Phase C migration helper — move a `core/X.py` to `src/lN_*/X.py`
and leave a backward-compat shim at the original path.

Usage:
    python3 scripts/migrate_to_layer.py core/exchange_client.py l1_exchange

The shim at the old location:
  - Imports the real module from the new location
  - Emits a DeprecationWarning ONCE per module
  - Re-exports every public attribute via `globals().update(...)`
  - Preserves `is`-identity for `from core.X import Y` callers

After moving, run:
  - python3 -m pytest tests/                                    (smoke)
  - python3 scripts/check_layer_imports.py                      (CI guard)

This script is idempotent: if the destination already exists or the source
is already a shim, it bails out without touching anything.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

SHIM_TEMPLATE = '''\
"""Backward-compatibility shim — DEPRECATED.

This module was moved to ``{new_dotted}`` in Phase C of the
directory-vs-layering alignment. Please update imports::

    # OLD (works via this shim, will be removed in Phase D)
    from core.{stem} import X
    import core.{stem}

    # NEW (recommended)
    from {new_dotted} import X
    import {new_dotted} as {stem}

The shim re-exports the *same* module object's public attributes, so
``from core.{stem} import X`` continues to return identical objects
(``is``-equal) to ``from {new_dotted} import X``.
"""
import warnings as _warnings
from {new_dotted} import *  # noqa: F401,F403  (re-export __all__ if present)
from {new_dotted} import __dict__ as _real_dict

_warnings.warn(
    "Importing 'core.{stem}' is deprecated; use '{new_dotted}' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export every public attribute (covers the common case where the
# upstream module has no __all__ but exposes many top-level symbols).
globals().update({{k: v for k, v in _real_dict.items() if not k.startswith("_")}})
'''


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: migrate_to_layer.py <core/X.py> <lN_layer>", file=sys.stderr)
        return 2

    src_path = Path(argv[1])
    layer = argv[2]
    if not src_path.exists():
        print(f"❌ source not found: {src_path}", file=sys.stderr)
        return 1
    if not src_path.parts[0] == "core":
        print("❌ this script only migrates files out of core/", file=sys.stderr)
        return 1

    stem = src_path.stem
    dst_dir = Path("src") / layer
    if not dst_dir.exists():
        print(f"❌ destination layer does not exist: {dst_dir}", file=sys.stderr)
        return 1
    dst_path = dst_dir / src_path.name
    if dst_path.exists():
        print(f"⚠️  destination already exists, skipping: {dst_path}")
        return 0

    new_dotted = f"src.{layer}.{stem}"

    # 1) Move with git
    rc = subprocess.run(
        ["git", "mv", str(src_path), str(dst_path)],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        # fallback to plain mv (untracked file)
        shutil.move(str(src_path), str(dst_path))
    print(f"✅ moved   {src_path}  →  {dst_path}")

    # 2) Write shim at the old path
    shim_text = SHIM_TEMPLATE.format(new_dotted=new_dotted, stem=stem)
    src_path.write_text(shim_text)
    print(f"✅ shim    {src_path}  →  re-exports from {new_dotted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
