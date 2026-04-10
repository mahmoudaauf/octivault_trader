#!/usr/bin/env python3
"""
Type Checking Error Analysis & Remediation Tool
Generates prioritized list of type errors for fixing
"""

import subprocess
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def run_mypy(directories: List[str]) -> str:
    """Run mypy and capture output"""
    cmd = ["python3", "-m", "mypy"] + directories + [
        "--ignore-missing-imports",
        "--show-error-codes",
        "--no-error-summary"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr

def parse_mypy_output(output: str) -> Dict[str, List[str]]:
    """Parse mypy output and categorize errors"""
    errors_by_file = defaultdict(list)
    errors_by_type = defaultdict(int)
    
    for line in output.split('\n'):
        if ':' in line and ('error:' in line or 'note:' in line):
            parts = line.split(':')
            if len(parts) >= 2:
                filepath = parts[0]
                if 'error:' in line:
                    error_type = line.split('[')[-1].rstrip(']') if '[' in line else 'unknown'
                    errors_by_file[filepath].append(line)
                    errors_by_type[error_type] += 1
    
    return dict(errors_by_file), dict(errors_by_type)

def generate_report(errors_by_file: Dict, errors_by_type: Dict):
    """Generate remediation report"""
    report = []
    report.append("=" * 80)
    report.append("TYPE CHECKING ERROR ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary
    total_errors = sum(len(v) for v in errors_by_file.values())
    total_files = len(errors_by_file)
    
    report.append(f"Total Errors: {total_errors}")
    report.append(f"Files with Errors: {total_files}")
    report.append(f"Average Errors per File: {total_errors/total_files:.1f}")
    report.append("")
    
    # Error Types
    report.append("TOP ERROR TYPES:")
    report.append("-" * 80)
    for error_type, count in sorted(errors_by_type.items(), key=lambda x: -x[1])[:10]:
        report.append(f"  [{error_type}]: {count} occurrences")
    report.append("")
    
    # Files by Error Count (Priority Order)
    report.append("FILES BY ERROR COUNT (Fix Priority):")
    report.append("-" * 80)
    sorted_files = sorted(errors_by_file.items(), key=lambda x: -len(x[1]))
    
    critical = []
    high = []
    medium = []
    low = []
    
    for filepath, errors in sorted_files:
        error_count = len(errors)
        filename = Path(filepath).name
        
        if error_count > 30:
            critical.append((filename, error_count))
        elif error_count > 15:
            high.append((filename, error_count))
        elif error_count > 5:
            medium.append((filename, error_count))
        else:
            low.append((filename, error_count))
    
    # Print by priority
    if critical:
        report.append("🔴 CRITICAL (>30 errors - fix first):")
        for name, count in critical:
            report.append(f"   {name}: {count} errors")
        report.append("")
    
    if high:
        report.append("🟠 HIGH (16-30 errors):")
        for name, count in high:
            report.append(f"   {name}: {count} errors")
        report.append("")
    
    if medium:
        report.append("🟡 MEDIUM (6-15 errors):")
        for name, count in medium[:10]:  # Show top 10
            report.append(f"   {name}: {count} errors")
        if len(medium) > 10:
            report.append(f"   ... and {len(medium)-10} more files")
        report.append("")
    
    if low:
        report.append(f"🟢 LOW (<5 errors): {len(low)} files")
        report.append("")
    
    # Recommendations
    report.append("REMEDIATION RECOMMENDATIONS:")
    report.append("-" * 80)
    report.append("1. START WITH: Core trading modules")
    report.append("   - core/execution_manager.py")
    report.append("   - core/signal_manager.py")
    report.append("   - core/position_manager.py")
    report.append("")
    report.append("2. THEN: Supporting modules")
    report.append("   - core/market_data_feed.py")
    report.append("   - agents/ml_forecaster.py")
    report.append("")
    report.append("3. QUICK WINS: Low-error files")
    report.append("   - Fix all <5 error files first (quick completion)")
    report.append("   - Builds momentum and confidence")
    report.append("")
    report.append("COMMON FIX PATTERNS:")
    report.append("-" * 80)
    report.append("1. Add return type hints:")
    report.append("   def get_value(self) -> str:")
    report.append("")
    report.append("2. Add parameter types:")
    report.append("   def process(self, value: str) -> int:")
    report.append("")
    report.append("3. Add None checks:")
    report.append("   if value is not None:")
    report.append("       return value.strip()")
    report.append("")
    report.append("4. Import TypedDict for structures:")
    report.append("   from typing import TypedDict")
    report.append("")
    
    return "\n".join(report)

def main():
    print("🔍 Running Type Checking Analysis...")
    print("")
    
    directories = ["core", "agents", "utils", "models"]
    output = run_mypy(directories)
    
    errors_by_file, errors_by_type = parse_mypy_output(output)
    report = generate_report(errors_by_file, errors_by_type)
    
    print(report)
    
    # Save to file
    report_path = Path(".archived/status_reports/type_checking_analysis.txt")
    report_path.write_text(report)
    print(f"\n📁 Report saved to: {report_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
