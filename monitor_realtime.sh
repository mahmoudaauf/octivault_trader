#!/bin/bash
# Real-Time System Monitoring Dashboard
# Monitors: Symbols, Balances, Prices, NAV, Signals, Trades

set -e

LOG_FILE="/tmp/octivault_nav_fixed.log"
REFRESH_RATE=3

echo "🎯 OCTI AI TRADING BOT - REAL-TIME MONITORING DASHBOARD"
echo "========================================================"
echo ""
echo "Press Ctrl+C to exit"
echo ""

# Function to get latest NAV
get_nav() {
    grep "BalanceSync.*💰" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '\$[0-9.]+' | head -1 | sed 's/\$//'
}

# Function to get NAV trend (GROWING/DECAYING)
get_nav_trend() {
    grep "BalanceSync.*💰" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '📈|📉' | head -1
}

# Function to count accepted symbols
get_symbol_count() {
    grep -c "Symbol accepted:" "$LOG_FILE" 2>/dev/null | head -1 || echo "0"
}

# Function to count active signals
get_signal_count() {
    grep "Signal cached" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE "Total in cache=[0-9]+" | grep -oE "[0-9]+" || echo "0"
}

# Function to get latest prices for major symbols
get_latest_prices() {
    local symbols=("BTCUSDT" "ETHUSDT" "BNBUSDT" "SOLUSDT" "DOGEUSDT")

    echo "  📊 Latest Prices:"
    for sym in "${symbols[@]}"; do
        price=$(grep "\[DEBUG_MDF\] price update $sym = " "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" | tail -1)
        if [ -z "$price" ]; then
            price="—"
        fi
        printf "     %-12s $%-8s\n" "$sym:" "$price"
    done
}

# Function to get balance updates
get_balances() {
    echo "  💰 Account Balances:"
    # Get USDT balance
    usdt=$(grep "USDT.*free" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE "free=[0-9.]+")
    if [ -z "$usdt" ]; then
        usdt="—"
    fi
    printf "     %-12s %s\n" "USDT:" "$usdt"

    # Get BTC balance
    btc=$(grep "BTC.*free" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE "free=[0-9.]+")
    if [ -z "$btc" ]; then
        btc="—"
    fi
    printf "     %-12s %s\n" "BTC:" "$btc"

    # Get ETH balance
    eth=$(grep "ETH.*free" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE "free=[0-9.]+")
    if [ -z "$eth" ]; then
        eth="—"
    fi
    printf "     %-12s %s\n" "ETH:" "$eth"
}

# Function to get recent signals
get_signals() {
    echo "  📡 Recent Signals (last 5):"
    grep "Signal cached for" "$LOG_FILE" 2>/dev/null | tail -5 | sed 's/.*Signal cached for /     ✓ /' | sed 's/ from / ← /'
}

# Function to get NAV history (last 5 updates)
get_nav_history() {
    echo "  📈 NAV History (last 5 updates):"
    grep "BalanceSync.*💰" "$LOG_FILE" 2>/dev/null | tail -5 | awk '{
        # Extract NAV value
        match($0, /\$[0-9.]+/, nav_arr)
        nav = nav_arr[0]

        # Extract trend emoji
        trend = "—"
        if (match($0, /📈/)) trend = "📈"
        if (match($0, /📉/)) trend = "📉"

        # Extract delta
        match($0, /delta=\$([^/]+)/, delta_arr)
        delta = delta_arr[1]

        printf "     %s %s %s\n", nav, trend, delta
    }' | tail -5
}

# Function to get recent trades
get_trades() {
    echo "  🔄 Recent Trading Activity:"
    local trade_count=$(grep -c "TRADE_SUBMITTED" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "     Total trades submitted: $trade_count"

    # Get last trade
    local last_trade=$(grep "TRADE_" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE "symbol.*reason" | head -1)
    if [ -n "$last_trade" ]; then
        echo "     Last trade: $last_trade"
    else
        echo "     Last trade: No trades yet"
    fi
}

# Main loop
while true; do
    clear

    echo "🎯 OCTI AI TRADING BOT - REAL-TIME MONITORING DASHBOARD"
    echo "========================================================"
    echo "Last update: $(date '+%H:%M:%S') | Refresh rate: ${REFRESH_RATE}s"
    echo ""

    # System Status
    echo "📊 SYSTEM STATUS"
    echo "─────────────────"
    nav=$(get_nav)
    trend=$(get_nav_trend)
    symbol_count=$(get_symbol_count)
    signal_count=$(get_signal_count)

    echo "  NAV: \$$nav $trend"
    echo "  Active Symbols: $symbol_count"
    echo "  Cached Signals: $signal_count"
    echo ""

    # Market Data
    echo "🌐 MARKET DATA"
    echo "──────────────"
    get_latest_prices
    echo ""

    # Account Status
    echo "💼 ACCOUNT STATUS"
    echo "──────────────────"
    get_balances
    echo ""

    # Signal Activity
    echo "📡 SIGNAL ACTIVITY"
    echo "───────────────────"
    get_signals
    echo ""

    # NAV Tracking
    echo "📊 NAV TRACKING"
    echo "────────────────"
    get_nav_history
    echo ""

    # Trading Activity
    echo "🔄 TRADING ACTIVITY"
    echo "────────────────────"
    get_trades
    echo ""

    echo "─────────────────────────────────────────────────────"
    echo "💡 Legend: 📈 = Growing | 📉 = Decaying | ✓ = Accepted"
    echo "Press Ctrl+C to exit. Refreshing in ${REFRESH_RATE}s..."

    sleep $REFRESH_RATE
done
