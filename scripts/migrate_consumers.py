"""Phase D — rewrite `core.X` imports to canonical `src.lN_*.X` paths.

Reads `src/_layer_index.py::LAYER_MODULES` to build the rewrite map, then
walks every file passed on the CLI (or the default consumer list) and
rewrites:

    from core.X import Y         →  from src.lN_*.X import Y
    import core.X                →  import src.lN_*.X as X
    import core.X as Z           →  import src.lN_*.X as Z
    core.X                       →  (left untouched in expressions; only top-level imports rewritten)

Usage:
    python3 scripts/migrate_consumers.py FILE1.py FILE2.py ...
    python3 scripts/migrate_consumers.py --dry-run FILE1.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src._layer_index import LAYER_MODULES  # noqa: E402


def build_map() -> dict[str, str]:
    """core.X / utils.X short-name → canonical dotted path."""
    out: dict[str, str] = {}
    for layer, mods in LAYER_MODULES.items():
        for short, dotted in mods.items():
            # Only rewrite if the canonical path is in src.lN_*  (not still core.* or utils.*)
            if dotted.startswith("src.l"):
                out[f"core.{short}"] = dotted
    return out


REWRITES = build_map()
# Sort longest first so 'core.market_data_websocket' beats 'core.market_data'
SORTED_KEYS = sorted(REWRITES.keys(), key=len, reverse=True)


def rewrite_line(line: str) -> str:
    """Rewrite import lines. Leave non-import code alone.

    Handles BOTH module-scope (column 0) and indented (function-body /
    inside try/except) imports::

        from core.X import Y
            from core.X import Y      # indented inside def / try
    """
    stripped = line.lstrip()
    if not (stripped.startswith("from core.") or stripped.startswith("import core.")):
        return line
    for old in SORTED_KEYS:
        new = REWRITES[old]
        # `from core.X import Y` → `from <new> import Y`
        # Match exact module boundary (not core.Xfoo)
        pattern = rf"\b{re.escape(old)}\b"
        if re.search(pattern, line):
            line = re.sub(pattern, new, line)
            break  # one rewrite per line is enough
    return line


def rewrite_file(path: Path, dry_run: bool = False) -> int:
    text = path.read_text()
    new_lines = []
    changes = 0
    for line in text.splitlines(keepends=True):
        new = rewrite_line(line)
        if new != line:
            changes += 1
        new_lines.append(new)
    if changes and not dry_run:
        path.write_text("".join(new_lines))
    return changes


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    files = [a for a in argv if not a.startswith("--")]
    if not files:
        print("usage: migrate_consumers.py [--dry-run] FILE...", file=sys.stderr)
        return 2
    total = 0
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"  ⚠️  missing: {f}")
            continue
        n = rewrite_file(p, dry_run=dry)
        if n:
            print(f"  ✏️  {f}  ({n} line{'s' if n != 1 else ''})")
            total += n
    print(f"\n{'[DRY-RUN] ' if dry else ''}Total rewrites: {total} across {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
