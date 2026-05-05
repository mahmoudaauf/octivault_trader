#!/bin/bash
set -e
cd "$(git rev-parse --show-toplevel)"

ARCHIVE_DEST="_archive/docs/historical_incidents"
mkdir -p "$ARCHIVE_DEST"

echo "📦 Archiving historical .md files at root..."

# Keep these
keep_files=(
  "README.md"
  "ARCHAEOLOGY_REPORT.md"
  "ARCHITECTURE_REALITY.md"
  "RUNBOOK.md"
  "00_START_HERE.md"
)

# Get all .md at root
all_md=$(find . -maxdepth 1 -name "*.md" -type f)

moved=0
skipped=0

for file in $all_md; do
  filename=$(basename "$file")

  # Check if should keep
  should_keep=0
  for keep in "${keep_files[@]}"; do
    if [[ "$filename" == "$keep" ]]; then
      should_keep=1
      break
    fi
  done

  if [[ $should_keep -eq 1 ]]; then
    echo "  ⊘ keep: $filename"
    skipped=$((skipped+1))
  else
    if git mv "$file" "$ARCHIVE_DEST/$filename"; then
      moved=$((moved+1))
      echo "  ✓ moved: $filename"
    else
      echo "  ⚠ error: $filename"
    fi
  fi
done

echo ""
echo "✅ Moved: $moved .md files"
echo "⊘ Kept: $skipped .md files"
echo "  Destination: $ARCHIVE_DEST"
