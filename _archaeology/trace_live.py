"""Trace the REAL dependency closure from the production entry point.

Anything reachable from 🎯_MASTER_SYSTEM_ORCHESTRATOR.py is LIVE.
Anything not reachable is a candidate for quarantine.

This is static analysis (AST) - imports inside `if False:` or dynamic
imports won't be caught. Manual review still needed before deletion.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "_archive",
    "_archaeology",
    ".claude",
    "artifacts",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
}

ENTRY_POINTS = [
    "🎯_MASTER_SYSTEM_ORCHESTRATOR.py",
    "master_orchestrator.py",  # alias if exists
]


def all_py():
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


# Build module-name -> path map (try multiple keys per file)
def build_module_index():
    files = list(all_py())
    idx: dict[str, Path] = {}
    for p in files:
        rel = p.relative_to(ROOT)
        parts = rel.with_suffix("").parts
        # full dotted: src.l0_core.config
        idx.setdefault(".".join(parts), p)
        # last-segment fallback: config
        idx.setdefault(parts[-1], p)
        # stem variations for files at root
        idx.setdefault(p.stem, p)
    return idx, files


def imported_modules(p: Path) -> set[str]:
    try:
        tree = ast.parse(p.read_text(errors="ignore"))
    except Exception:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name)
                out.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod:
                out.add(mod)
                out.add(mod.split(".")[0])
                out.add(mod.split(".")[-1])
            for n in node.names:
                # `from x import y` - y might also be a module
                if mod:
                    out.add(f"{mod}.{n.name}")
                out.add(n.name)
    return out


def trace(entries: list[Path], idx: dict[str, Path]) -> set[Path]:
    reached: set[Path] = set()
    stack = list(entries)
    while stack:
        cur = stack.pop()
        if cur in reached:
            continue
        reached.add(cur)
        for name in imported_modules(cur):
            target = idx.get(name)
            if target and target not in reached:
                stack.append(target)
    return reached


def main():
    idx, files = build_module_index()
    entries = [ROOT / e for e in ENTRY_POINTS if (ROOT / e).exists()]
    if not entries:
        print("❌ No entry point found")
        sys.exit(1)
    print(f"Entry points: {[e.name for e in entries]}")

    reached = trace(entries, idx)
    all_set = set(files)
    unreached = all_set - reached

    rel = lambda p: str(p.relative_to(ROOT))
    reached_sorted = sorted(rel(p) for p in reached)
    unreached_sorted = sorted(rel(p) for p in unreached)

    (ROOT / "_archaeology" / "live_dependency_closure.txt").write_text("\n".join(reached_sorted))
    (ROOT / "_archaeology" / "unreached_from_entry.txt").write_text("\n".join(unreached_sorted))

    # Bucket the unreached
    buckets = {"src/": [], "agents/": [], "automation/": [], "root/": [], "other/": []}
    for path in unreached_sorted:
        if path.startswith("src/"):
            buckets["src/"].append(path)
        elif path.startswith("agents/"):
            buckets["agents/"].append(path)
        elif path.startswith("automation/"):
            buckets["automation/"].append(path)
        elif "/" not in path:
            buckets["root/"].append(path)
        else:
            buckets["other/"].append(path)

    print(f"\nTotal py files (excl .claude): {len(files)}")
    print(f"  ✅ Reachable from entry:  {len(reached)}  ({100*len(reached)//len(files)}%)")
    print(f"  ⚠️  NOT reachable:        {len(unreached)}  ({100*len(unreached)//len(files)}%)")
    print("\nUnreachable by location:")
    for k, v in buckets.items():
        print(f"  {k:<14} {len(v)}")
    print("\nDetails written to:")
    print("  _archaeology/live_dependency_closure.txt")
    print("  _archaeology/unreached_from_entry.txt")


if __name__ == "__main__":
    main()
