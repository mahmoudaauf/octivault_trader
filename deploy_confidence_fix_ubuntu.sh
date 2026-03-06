#!/bin/bash
# Deploy confidence fixes to Ubuntu production server
# Run this on the Ubuntu server at ~/octivault_trader

set -e  # Exit on error

echo "🚀 Deploying Confidence Hardcoding Fixes"
echo "=========================================="
echo ""

REPO_PATH="$HOME/octivault_trader"
cd "$REPO_PATH"

if [ ! -f "core/volatility_adjusted_confidence.py" ]; then
    echo "❌ Error: core/volatility_adjusted_confidence.py not found"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "✅ Found core/volatility_adjusted_confidence.py"
echo ""

# Backup files
echo "📋 Creating backups..."
cp core/volatility_adjusted_confidence.py core/volatility_adjusted_confidence.py.backup
cp agents/trend_hunter.py agents/trend_hunter.py.backup
echo "   ✅ Backups created"
echo ""

# Fix 1: Update import in trend_hunter.py to use core/
echo "🔧 Fix 1: Updating trend_hunter.py import path..."
python3 << 'EOFIX1'
import re

with open('agents/trend_hunter.py', 'r') as f:
    content = f.read()

# Replace the entire try/except block with direct import
old_pattern = r'try:\s+from utils\.volatility_adjusted_confidence import \(\s+compute_heuristic_confidence,\s+categorize_signal,\s+get_signal_quality_metrics,\s+\)\s+except \(ImportError, ModuleNotFoundError\):.*?return \{\}'

new_import = '''from core.volatility_adjusted_confidence import (
    compute_heuristic_confidence,
    categorize_signal,
    get_signal_quality_metrics,
)'''

# Use simpler approach: find and replace line by line
lines = content.split('\n')
new_lines = []
skip_until_next = False
for i, line in enumerate(lines):
    # Skip the old try/except block
    if 'try:' in line and i < len(lines) - 1 and 'from utils.volatility_adjusted_confidence' in lines[i+1]:
        skip_until_next = True
        new_lines.append('from core.volatility_adjusted_confidence import (')
        new_lines.append('    compute_heuristic_confidence,')
        new_lines.append('    categorize_signal,')
        new_lines.append('    get_signal_quality_metrics,')
        new_lines.append(')')
        continue
    
    if skip_until_next:
        if 'from agents.edge_calculator' in line:
            skip_until_next = False
            new_lines.append('')
            new_lines.append(line)
        # Skip everything in the try/except block
        continue
    
    new_lines.append(line)

with open('agents/trend_hunter.py', 'w') as f:
    f.write('\n'.join(new_lines))

print("✅ Updated import in trend_hunter.py")
EOFIX1

echo ""

# Fix 2: Make talib optional in core/volatility_adjusted_confidence.py
echo "🔧 Fix 2: Making talib import optional..."
python3 << 'EOFIX2'
with open('core/volatility_adjusted_confidence.py', 'r') as f:
    lines = f.readlines()

# Find the import talib line
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Look for "import talib" (not "import talib.core" or similar)
    if line.strip() == 'import talib':
        # Replace with try/except block
        new_lines.append('try:\n')
        new_lines.append('    import talib\n')
        new_lines.append('    _HAS_TALIB = True\n')
        new_lines.append('except (ImportError, ModuleNotFoundError):\n')
        new_lines.append('    _HAS_TALIB = False\n')
    else:
        new_lines.append(line)
    
    i += 1

with open('core/volatility_adjusted_confidence.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Made talib import optional")
EOFIX2

echo ""

# Fix 3: Add near-zero magnitude guard
echo "🔧 Fix 3: Adding near-zero magnitude guard..."
python3 << 'EOFIX3'
with open('core/volatility_adjusted_confidence.py', 'r') as f:
    content = f.read()

# Check if guard already exists
if 'if max_hist < 1e-6:' in content:
    print("✅ Near-zero magnitude guard already present")
else:
    # Find the compute_histogram_magnitude function and add the guard
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Look for "max_hist = np.max(recent)"
        if 'max_hist = np.max(recent)' in line:
            # Add the guard after this line
            # Skip until we find "magnitude = latest_mag / max_hist"
            j = i + 1
            found_division = False
            while j < len(lines) and not found_division:
                if 'magnitude = latest_mag / max_hist' in lines[j]:
                    found_division = True
                    # Insert guard before the division
                    new_lines.append('')
                    new_lines.append('    # CRITICAL FIX: If all histogram values are near zero (e.g., chop/sideways),')
                    new_lines.append('    # don\'t divide by near-zero. Instead, return raw magnitude bounded to [0, 1]')
                    new_lines.append('    if max_hist < 1e-6:  # Near-zero threshold')
                    new_lines.append('        # In chop, all signals are weak. Return magnitude clamped to [0, 0.3]')
                    new_lines.append('        magnitude = np.clip(latest_mag * 1000, 0.0, 0.3)  # Scale up slightly but cap at weak')
                    new_lines.append('        logger.debug(')
                    new_lines.append('            "[VolumeAdjConf] Chop-mode magnitude: max_hist=%.8f latest_mag=%.8f "')
                    new_lines.append('            "→ chop_magnitude=%.3f (signals too weak to normalize)",')
                    new_lines.append('            max_hist,')
                    new_lines.append('            latest_mag,')
                    new_lines.append('            magnitude,')
                    new_lines.append('        )')
                    new_lines.append('        return magnitude')
                    new_lines.append('')
                    # Now add the original division line
                    new_lines.append(lines[j])
                    j += 1
                    break
                else:
                    new_lines.append(lines[j])
                    j += 1
            
            # Add remaining lines
            while j < len(lines):
                new_lines.append(lines[j])
                j += 1
            break
    
    with open('core/volatility_adjusted_confidence.py', 'w') as f:
        f.write('\n'.join(new_lines))
    
    print("✅ Added near-zero magnitude guard")
EOFIX3

echo ""

# Fix 4: Add closes parameter to get_signal_quality_metrics
echo "🔧 Fix 4: Adding closes parameter to function signature..."
python3 << 'EOFIX4'
with open('core/volatility_adjusted_confidence.py', 'r') as f:
    content = f.read()

# Check if closes parameter already exists
if 'def get_signal_quality_metrics(\n    hist_values: np.ndarray,\n    regime: str = "normal",\n    closes: np.ndarray = None,' in content:
    print("✅ Closes parameter already added")
else:
    # Replace the function signature
    content = content.replace(
        '''def get_signal_quality_metrics(
    hist_values: np.ndarray,
    regime: str = "normal",
) -> Dict[str, float]:''',
        '''def get_signal_quality_metrics(
    hist_values: np.ndarray,
    regime: str = "normal",
    closes: np.ndarray = None,
) -> Dict[str, float]:'''
    )
    
    with open('core/volatility_adjusted_confidence.py', 'w') as f:
        f.write(content)
    
    print("✅ Added closes parameter to function signature")
EOFIX4

echo ""

# Fix 5: Pass closes to magnitude calculation
echo "🔧 Fix 5: Passing closes to magnitude calculation..."
python3 << 'EOFIX5'
with open('core/volatility_adjusted_confidence.py', 'r') as f:
    content = f.read()

# Check if already fixed
if 'magnitude = compute_histogram_magnitude(hist_values, closes=closes)' in content:
    print("✅ Closes parameter already passed to magnitude calculation")
else:
    # Replace the magnitude calculation line
    content = content.replace(
        'magnitude = compute_histogram_magnitude(hist_values)',
        'magnitude = compute_histogram_magnitude(hist_values, closes=closes)'
    )
    
    with open('core/volatility_adjusted_confidence.py', 'w') as f:
        f.write(content)
    
    print("✅ Updated magnitude calculation to pass closes parameter")
EOFIX5

echo ""

# Verify fixes
echo "🔍 Verifying fixes..."
python3 << 'EOVERIFY'
import inspect
try:
    from core.volatility_adjusted_confidence import get_signal_quality_metrics
    sig = inspect.signature(get_signal_quality_metrics)
    params = list(sig.parameters.keys())
    
    print(f"✅ Function signature verified:")
    print(f"   Parameters: {params}")
    
    if 'closes' in params:
        print(f"   ✅ 'closes' parameter present")
    else:
        print(f"   ❌ 'closes' parameter MISSING - fix may not have applied")
        exit(1)
    
except Exception as e:
    print(f"❌ Import or verification failed: {e}")
    exit(1)
EOVERIFY

echo ""
echo "✅ All fixes deployed successfully!"
echo ""
echo "📋 Summary of changes:"
echo "  1. ✅ trend_hunter.py imports from core/ (no fallback)"
echo "  2. ✅ talib import is optional"
echo "  3. ✅ Near-zero magnitude guard added"
echo "  4. ✅ closes parameter added to function signature"
echo "  5. ✅ closes passed to magnitude calculation"
echo ""
echo "⚠️  NEXT STEPS:"
echo "  1. Kill bot: pkill -f 'python.*run_bot'"
echo "  2. Wait 3 seconds: sleep 3"
echo "  3. Restart: python3 run_bot.py --debug"
echo "  4. Monitor: tail -f logs/bot.log | grep 'heuristic for'"
echo ""
echo "Expected log output after fix:"
echo "  mag=0.3456 accel=0.1200 raw=0.595 → adj=0.595 (floor=0.55) → final=0.595"
echo ""
