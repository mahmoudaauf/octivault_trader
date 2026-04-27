#!/usr/bin/env python3
"""
BALANCE MONITORING DASHBOARD
Real-time display of portfolio performance metrics
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent))

from system_state_manager import SystemStateManager


class BalanceDashboard:
    """Real-time balance and performance dashboard"""
    
    def __init__(self):
        self.state_mgr = SystemStateManager()
        self.last_balance = None
        self.last_update = None
        
    def clear_screen(self):
        """Clear terminal screen"""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def get_balance_data(self):
        """Get current balance data from state"""
        try:
            ctx = self.state_mgr.get_system_context()
            return ctx.get('system_status', {}).get('balance_tracking', {})
        except:
            return {}
    
    def display_header(self):
        """Display dashboard header"""
        print("╔════════════════════════════════════════════════════════════════════════════════╗")
        print("║                      💰 BALANCE MONITORING DASHBOARD 💰                        ║")
        print("║                     Real-Time Portfolio Performance Tracking                  ║")
        print("╚════════════════════════════════════════════════════════════════════════════════╝")
        print()
    
    def display_performance_metrics(self, balance_data):
        """Display current performance metrics"""
        if not balance_data:
            print("⏳ Waiting for balance data...")
            return
        
        current = balance_data.get('current_balance', 0)
        initial = balance_data.get('initial_balance', 0)
        peak = balance_data.get('peak_balance', 0)
        lowest = balance_data.get('lowest_balance', 0)
        
        change = current - initial
        change_pct = (change / initial * 100) if initial > 0 else 0
        peak_pct = ((peak - initial) / initial * 100) if initial > 0 else 0
        low_pct = ((lowest - initial) / initial * 100) if initial > 0 else 0
        
        # Determine status color/emoji
        if change > 0:
            status_emoji = "📈"
            status_text = "GAINING"
        elif change < 0:
            status_emoji = "📉"
            status_text = "LOSING"
        else:
            status_emoji = "➡️"
            status_text = "STABLE"
        
        # Display metrics
        print("┌─ CURRENT BALANCE ─────────────────────────────────────────────────────────────┐")
        print(f"│ ${current:,.2f}                                                                    │")
        print(f"│ {status_emoji} {status_text} - Change: {change:+.2f} ({change_pct:+.2f}%)                                     │")
        print("└───────────────────────────────────────────────────────────────────────────────┘")
        print()
        
        print("┌─ PERFORMANCE SUMMARY ─────────────────────────────────────────────────────────┐")
        print(f"│ Initial Balance:  ${initial:>12,.2f}                                          │")
        print(f"│ Peak Balance:     ${peak:>12,.2f}   ({peak_pct:>+6.2f}%)                       │")
        print(f"│ Lowest Balance:   ${lowest:>12,.2f}   ({low_pct:>+6.2f}%)                       │")
        print(f"│ Current Balance:  ${current:>12,.2f}   ({change_pct:>+6.2f}%)                       │")
        print("└───────────────────────────────────────────────────────────────────────────────┘")
        print()
    
    def display_updates(self, balance_data):
        """Display update frequency and history"""
        updates = balance_data.get('history_length', 0)
        last_update = balance_data.get('last_update', 'Never')
        
        print("┌─ UPDATE FREQUENCY ────────────────────────────────────────────────────────────┐")
        print(f"│ Total Updates: {updates:>4}                                                  │")
        print(f"│ Last Update:   {last_update}                                 │")
        print("└───────────────────────────────────────────────────────────────────────────────┘")
        print()
    
    def display_system_status(self):
        """Display system status"""
        try:
            ctx = self.state_mgr.get_system_context()
            status = ctx.get('system_status', {})
            phase = status.get('current_phase', 'Unknown')
            task = status.get('current_task', 'Unknown')
            recovery = "✅ ENABLED" if status.get('recovery_enabled') else "❌ DISABLED"
            
            print("┌─ SYSTEM STATUS ───────────────────────────────────────────────────────────┐")
            print(f"│ Phase: {phase:<20}                                                    │")
            print(f"│ Task:  {task:<20}                                                    │")
            print(f"│ Auto-Recovery: {recovery}                                                │")
            print("└───────────────────────────────────────────────────────────────────────────┘")
            print()
        except Exception as e:
            print(f"⚠️  Could not retrieve system status: {e}\n")
    
    def display_dashboard(self):
        """Display complete dashboard"""
        self.clear_screen()
        self.display_header()
        
        balance_data = self.get_balance_data()
        self.display_performance_metrics(balance_data)
        self.display_updates(balance_data)
        self.display_system_status()
        
        # Display refresh info
        print("📊 Dashboard refreshing every 5 seconds (Press Ctrl+C to exit)")
        print(f"⏰ Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run(self, refresh_interval=5):
        """Run continuous dashboard"""
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\n✅ Dashboard stopped")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    dashboard = BalanceDashboard()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Single display mode (no refresh)
        dashboard.display_dashboard()
    else:
        # Continuous refresh mode (default)
        dashboard.run(refresh_interval=5)
