#!/bin/bash
# ============================================================================
# SYNC_TO_SERVER.sh - Synchronize corrected files to Ubuntu server
# ============================================================================
# Usage: ./SYNC_TO_SERVER.sh ubuntu@ip-172-31-37-246
# ============================================================================

set -e

if [ $# -ne 1 ]; then
    echo "Usage: ./SYNC_TO_SERVER.sh <user@host>"
    echo "Example: ./SYNC_TO_SERVER.sh ubuntu@ip-172-31-37-246"
    exit 1
fi

SERVER="$1"
LOCAL_DIR="/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                 SYNCING CORRECTED FILES TO SERVER                       ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Server: $SERVER"
echo "Local dir: $LOCAL_DIR"
echo ""

FILES=(
    "core/execution_manager.py"
    "core/shared_state.py"
    "core/meta_controller.py"
)

for FILE in "${FILES[@]}"; do
    echo "Syncing: $FILE"
    scp "$LOCAL_DIR/$FILE" "$SERVER:~/octivault_trader/$FILE"
    echo "  ✅ $FILE synced"
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                         VERIFICATION                                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

ssh "$SERVER" << 'VERIFY'
cd ~/octivault_trader

echo "Checking syntax on server..."
python3 -c "
import sys
modules = ['core.execution_manager', 'core.shared_state', 'core.meta_controller']
for m in modules:
    try:
        __import__(m)
        print(f'✅ {m}')
    except Exception as e:
        print(f'❌ {m}: {e}')
        sys.exit(1)
print('')
print('All modules import successfully!')
"

VERIFY

echo ""
echo "✅ Sync and verification complete!"
echo ""
echo "Next step: Restart Phase 9"
echo "  ssh $SERVER"
echo "  cd ~/octivault_trader"
echo "  nohup python3 -u main_phased.py --phase 9 > logs/clean_run.log 2>&1 &"
echo ""
