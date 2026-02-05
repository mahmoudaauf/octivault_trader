import os
import re

TARGET_DIRS = ['agents', 'core']
SIMULATION_FLAGS = [
    r'SIMULATION_MODE\s*=\s*True',
    r'if\s+config\.SIMULATION_MODE',
    r'config\.SIMULATION_MODE\s*==\s*True',
    r'ExecutionManager\(.*simulate.*\)',
    r'\.get\([\'"]SIMULATION_MODE[\'"]',
]

DUMMY_PATTERNS = [
    r'dummy prediction',
    r'print\(.*dummy.*\)',
    r'return ["\'](buy|sell|hold)["\']',
]

MOCK_DATA_PATTERNS = [
    r'import random',
    r'random\.choice',
    r'fetch_dummy_data',
    r'ohlcv = \[.*\]',  # likely mock candles
]

NO_SIMULATION_CHECK = [
    r'ExecutionManager.*place_order\(.*\)',  # to check for lack of live/simulate branching
]

def audit_file(filepath):
    issues = []
    with open(filepath, 'r') as file:
        content = file.read()
        for flag in SIMULATION_FLAGS:
            if re.search(flag, content):
                issues.append(f"⚠️ Simulation flag detected: {flag}")
        for pattern in DUMMY_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"❌ Dummy logic detected: {pattern}")
        for pattern in MOCK_DATA_PATTERNS:
            if re.search(pattern, content):
                issues.append(f"🚫 Mock data usage: {pattern}")
        for pattern in NO_SIMULATION_CHECK:
            matches = re.findall(pattern, content)
            for m in matches:
                if "live=" not in m:
                    issues.append(f"⚠️ Potential live flag missing in: {m.strip()}")
    return issues

def run_audit():
    print("📊 Running Live Readiness Audit...\n")
    total_issues = 0
    for dirpath in TARGET_DIRS:
        for root, _, files in os.walk(dirpath):
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    issues = audit_file(path)
                    if issues:
                        print(f"\n🔎 {path}")
                        for issue in issues:
                            print(f"  {issue}")
                            total_issues += 1
    if total_issues == 0:
        print("✅ All modules look ready for live trading!")
    else:
        print(f"\n🚨 Total issues found: {total_issues}")

if __name__ == "__main__":
    run_audit()
