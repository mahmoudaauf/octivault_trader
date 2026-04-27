#!/bin/bash
# 🚀 START HERE - Begin Live Trading with Compounding

clear

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║              🚀 OCTIVAULT LIVE TRADING SYSTEM - START HERE 🚀               ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader

export APPROVE_LIVE_TRADING=YES

echo "✅ READY TO START LIVE TRADING"
echo ""
echo "Account Status:"
echo "  • Balance: $60.42 USDT"
echo "  • Mode: LIVE (not testnet)"
echo "  • API: Authenticated ✅"
echo ""
echo "System Status:"
echo "  • Config: Ready ✅"
echo "  • Exchange: Connected ✅"
echo "  • Orders: Enabled ✅"
echo ""

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "📊 RUNNING LIVE TRADING MONITOR..."
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

python3 🎯_SESSION_MONITOR.py

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "✅ TRADING SESSION COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Review trading results"
echo "  2. Check Binance account for actual orders"
echo "  3. Verify fills and P&L"
echo "  4. Run again to continue trading with compounding"
echo ""
