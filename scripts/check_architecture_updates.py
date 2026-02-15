#!/usr/bin/env python3
"""
Architecture Documentation Maintenance Helper

This script helps maintain the ARCHITECTURE.md file by checking for recent changes
to core components and reminding developers to update documentation.

Usage:
    python scripts/check_architecture_updates.py [--days 7] [--verbose]

Checks for changes in the last N days to core architecture files and reports
whether the ARCHITECTURE.md file has been updated accordingly.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Core architecture files that should trigger documentation updates
CORE_ARCHITECTURE_FILES = [
    "core/meta_controller.py",
    "core/execution_manager.py",
    "core/shared_state.py",
    "core/tp_sl_engine.py",
    "core/agent_manager.py",
    "core/risk_manager.py",
    "core/portfolio_manager.py",
    "agents/",  # Any agent changes
    "ARCHITECTURE.md"
]

def run_git_command(cmd):
    """Run a git command and return the output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)

def get_recent_changes(days=7):
    """Get files changed in the last N days."""
    since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    success, output = run_git_command(f"git log --since='{since_date}' --name-only --pretty=format:")

    if not success:
        print(f"Error running git log: {output}")
        return set()

    changed_files = set()
    for line in output.split('\n'):
        line = line.strip()
        if line and not line.startswith(' ') and line != '':
            changed_files.add(line)

    return changed_files

def check_architecture_updates(changed_files, verbose=False):
    """Check if architecture documentation needs updating."""
    core_changes = []
    arch_updated = False

    for file_path in changed_files:
        # Check if it's a core architecture file
        for core_file in CORE_ARCHITECTURE_FILES:
            if file_path.startswith(core_file) or core_file in file_path:
                if file_path == "ARCHITECTURE.md":
                    arch_updated = True
                else:
                    core_changes.append(file_path)
                break

    if verbose:
        print(f"Found {len(changed_files)} total changed files")
        print(f"Found {len(core_changes)} core architecture changes")

    if core_changes and not arch_updated:
        print("⚠️  ARCHITECTURE UPDATE REQUIRED")
        print("The following core files have changed but ARCHITECTURE.md hasn't been updated:")
        for file in core_changes:
            print(f"  - {file}")
        print("\nPlease update ARCHITECTURE.md to reflect these changes.")
        print("See the 'Maintenance Guidelines' section in ARCHITECTURE.md for details.")
        return False
    elif core_changes and arch_updated:
        print("✅ Architecture documentation appears to be up to date.")
        return True
    else:
        print("ℹ️  No core architecture changes detected in the specified time period.")
        return True

def main():
    parser = argparse.ArgumentParser(description="Check if architecture documentation needs updating")
    parser.add_argument("--days", type=int, default=7, help="Number of days to check for changes (default: 7)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Check if we're in a git repository
    success, _ = run_git_command("git rev-parse --git-dir")
    if not success:
        print("Error: Not in a git repository")
        sys.exit(1)

    print(f"Checking for architecture changes in the last {args.days} days...")

    changed_files = get_recent_changes(args.days)
    if not changed_files:
        print("No file changes found in the specified period.")
        return

    success = check_architecture_updates(changed_files, args.verbose)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()