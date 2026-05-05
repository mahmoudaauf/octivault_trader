"""Generate ARCHITECTURE_REALITY.md - read-only inventory of the codebase.

Heuristics:
- ENTRY-POINT: has `if __name__ == "__main__"` AND not imported by anything
- LIBRARY:    imported by >=1 other file
- ORPHAN:     not imported, no __main__   -> quarantine candidate
- PATCH-ARTIFACT: filename screams "fix/old/backup/v2"
- SUSPECT-DUPLICATE: another file has near-identical name (e.g. capital_allocator.py vs CAPITAL_ALLOCATOR_FIX_CODE.py)
"""
from __future__ import annotations
import ast, hashlib, re, subprocess
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "node_modules",
             "_archive", "_archaeology", ".mypy_cache", ".pytest_cache",
             "artifacts", ".ipynb_checkpoints", ".claude", "build", "dist",
             ".tox", ".eggs", "site-packages"}

PATCH_RE = re.compile(r"(?i)(_fix|fix_|_old|old_|_backup|backup_|_deprecated|"
                      r"_v[0-9]+|_new|_temp|temp_|_copy|_bak)")

def py_files():
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p

def has_main(text: str) -> bool:
    return bool(re.search(r'__name__\s*==\s*["\']__main__["\']', text))

def imports_in(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except Exception:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # take last component too for resolution against stems
                parts = node.module.split(".")
                out.add(parts[0])
                out.add(parts[-1])
            for n in node.names:
                out.add(n.name)
    return out

def git_mtime(p: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "log", "-1", "--format=%ai", "--", str(p.relative_to(ROOT))],
            cwd=ROOT, capture_output=True, text=True, timeout=5)
        return (r.stdout.strip() or "untracked")[:10]
    except Exception:
        return "n/a"

def main():
    files = sorted(py_files())
    texts = {p: p.read_text(errors="ignore") for p in files}

    # Build stem index for crude resolution
    stem_to_files: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        stem_to_files[p.stem].append(p)

    importers: dict[Path, set[Path]] = {p: set() for p in files}
    for p, txt in texts.items():
        for name in imports_in(txt):
            for target in stem_to_files.get(name, []):
                if target != p:
                    importers[target].add(p)

    rows = []
    for p in files:
        txt = texts[p]
        rows.append({
            "path": str(p.relative_to(ROOT)),
            "name": p.name,
            "loc": txt.count("\n") + 1,
            "size": p.stat().st_size,
            "has_main": has_main(txt),
            "imported_by": len(importers[p]),
            "importer_list": sorted(str(x.relative_to(ROOT)) for x in importers[p])[:5],
            "git_mtime": git_mtime(p),
            "is_patch_named": bool(PATCH_RE.search(p.name)) or p.name.isupper().__bool__() and "FIX" in p.name.upper(),
            "screams_fix": "FIX" in p.name.upper() or "BACKUP" in p.name.upper() or "OLD" in p.name.upper(),
            "in_src": p.parts[0] == "src" if len(p.parts) > 1 else False,
            "at_root": p.parent == ROOT,
        })

    # Detect near-duplicates by lowercased stem matching
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    norm_groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = norm(Path(r["path"]).stem.replace("fix", "").replace("backup", "")
                   .replace("old", "").replace("new", "").replace("copy", ""))
        if key:
            norm_groups[key].append(r)
    duplicates = {k: v for k, v in norm_groups.items() if len(v) > 1}

    def status(r):
        if r["screams_fix"]:                       return "🟠 PATCH-ARTIFACT"
        if r["imported_by"] == 0 and not r["has_main"]:
            return "🔴 ORPHAN"
        if r["has_main"] and r["imported_by"] == 0:
            return "🟢 ENTRY-POINT"
        if r["imported_by"] > 0:                   return "🟢 LIBRARY"
        return "⚪ UNKNOWN"

    for r in rows:
        r["status"] = status(r)

    # ---------- Write report ----------
    out = []
    out.append("# ARCHITECTURE REALITY")
    out.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}_  ")
    out.append(f"_Total Python files scanned: **{len(rows)}**_")
    out.append("")
    out.append("## 1. Status summary")
    out.append("")
    out.append("| Status | Count |")
    out.append("|---|---|")
    for k, v in Counter(r["status"] for r in rows).most_common():
        out.append(f"| {k} | {v} |")
    out.append("")

    out.append("## 2. Location summary")
    out.append("")
    out.append("| Location | Count |")
    out.append("|---|---|")
    out.append(f"| `src/` (clean architecture) | {sum(1 for r in rows if r['in_src'])} |")
    out.append(f"| Repo root (likely chaos)    | {sum(1 for r in rows if r['at_root'])} |")
    out.append(f"| Other folders               | {sum(1 for r in rows if not r['in_src'] and not r['at_root'])} |")
    out.append("")

    out.append("## 3. 🟢 Entry-point candidates (has `__main__`, not imported)")
    out.append("")
    eps = [r for r in rows if r["status"] == "🟢 ENTRY-POINT"]
    eps.sort(key=lambda r: (-r["loc"], r["path"]))
    out.append("| File | LOC | Last commit |")
    out.append("|---|---:|---|")
    for r in eps[:40]:
        out.append(f"| `{r['path']}` | {r['loc']} | {r['git_mtime']} |")
    if len(eps) > 40:
        out.append(f"| _...{len(eps)-40} more_ | | |")
    out.append("")

    out.append("## 4. 🟠 Patch-artifacts (filename contains FIX/OLD/BACKUP/v2/etc.)")
    out.append("")
    pa = [r for r in rows if r["status"] == "🟠 PATCH-ARTIFACT"]
    pa.sort(key=lambda r: r["path"])
    out.append("| File | LOC | Imported by | Last commit |")
    out.append("|---|---:|---:|---|")
    for r in pa:
        out.append(f"| `{r['path']}` | {r['loc']} | {r['imported_by']} | {r['git_mtime']} |")
    out.append("")

    out.append("## 5. 🔴 Orphans (no importers, no `__main__`) — top quarantine candidates")
    out.append("")
    orphans = [r for r in rows if r["status"] == "🔴 ORPHAN"]
    orphans.sort(key=lambda r: (-r["loc"], r["path"]))
    out.append(f"_Total orphans: **{len(orphans)}**_")
    out.append("")
    out.append("| File | LOC | Last commit |")
    out.append("|---|---:|---|")
    for r in orphans[:60]:
        out.append(f"| `{r['path']}` | {r['loc']} | {r['git_mtime']} |")
    if len(orphans) > 60:
        out.append(f"| _...{len(orphans)-60} more (see orphans_full.txt)_ | | |")
    out.append("")

    out.append("## 6. Suspected duplicate groups (similar normalized names)")
    out.append("")
    if not duplicates:
        out.append("_None detected._")
    else:
        for key, group in sorted(duplicates.items(), key=lambda kv: -len(kv[1]))[:30]:
            if len(group) < 2:
                continue
            out.append(f"### Group `{key}`")
            out.append("")
            for r in sorted(group, key=lambda x: x["path"]):
                out.append(f"- `{r['path']}` — {r['loc']} LOC, {r['status']}, imported_by={r['imported_by']}")
            out.append("")

    out.append("## 7. Top 20 LIBRARIES by fan-in (most-imported = most critical)")
    out.append("")
    libs = sorted([r for r in rows if r["status"] == "🟢 LIBRARY"],
                  key=lambda r: -r["imported_by"])
    out.append("| File | Imported by N files | Sample importers |")
    out.append("|---|---:|---|")
    for r in libs[:20]:
        sample = ", ".join(f"`{x}`" for x in r["importer_list"][:3])
        out.append(f"| `{r['path']}` | {r['imported_by']} | {sample} |")
    out.append("")

    (ROOT / "ARCHITECTURE_REALITY.md").write_text("\n".join(out))

    # Full orphan list for quarantine planning
    (ROOT / "_archaeology" / "orphans_full.txt").write_text(
        "\n".join(r["path"] for r in orphans))
    (ROOT / "_archaeology" / "patch_artifacts.txt").write_text(
        "\n".join(r["path"] for r in pa))
    (ROOT / "_archaeology" / "entry_points.txt").write_text(
        "\n".join(r["path"] for r in eps))

    print(f"✅ Wrote ARCHITECTURE_REALITY.md")
    print(f"   Files scanned: {len(rows)}")
    print(f"   Entry points: {len(eps)} | Libraries: {len(libs)} | "
          f"Orphans: {len(orphans)} | Patch-artifacts: {len(pa)}")
    print(f"   Suspected duplicate groups: {sum(1 for v in duplicates.values() if len(v)>1)}")

if __name__ == "__main__":
    main()
