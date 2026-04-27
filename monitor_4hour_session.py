#!/usr/bin/env python3
"""
4-HOUR SESSION MONITOR DASHBOARD
Real-time monitoring of 4-hour session with checkpoint tracking
Dynamic balance classification for any account size
"""

import json
import time
from datetime import datetime
from pathlib import Path
import os
import sys

# Import dynamic threshold calculator
try:
    from balance_threshold_config import DynamicBalanceThresholds
except ImportError:
    DynamicBalanceThresholds = None


class FourHourSessionMonitor:
    """Monitor 4-hour trading session in real-time"""
    
    def __init__(self):
        self.state_dir = Path("state")
        self.refresh_interval = 5  # seconds
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def load_session_state(self):
        """Load current session state"""
        session_file = self.state_dir / "4hour_session_state.json"
        if session_file.exists():
            with open(session_file, "r") as f:
                return json.load(f)
        return None
    
    def load_latest_checkpoint(self):
        """Load latest checkpoint"""
        checkpoint_files = sorted(self.state_dir.glob("checkpoint_*.json"))
        if checkpoint_files:
            with open(checkpoint_files[-1], "r") as f:
                return json.load(f)
        return None
    
    def load_final_state(self):
        """Load final session state"""
        final_file = self.state_dir / "4hour_session_final.json"
        if final_file.exists():
            with open(final_file, "r") as f:
                return json.load(f)
        return None
    
    def get_all_checkpoints(self):
        """Get all checkpoint files"""
        return sorted(self.state_dir.glob("checkpoint_*.json"))
    
    def format_duration(self, minutes):
        """Format duration as HH:MM"""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}m"
    
    def display_header(self):
        """Display dashboard header"""
        print("╔" + "=" * 78 + "╗")
        print("║" + "4-HOUR SESSION MONITOR DASHBOARD".center(78) + "║")
        print("║" + f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(78) + "║")
        print("╚" + "=" * 78 + "╝")
        print()
    
    def display_session_info(self, session_state):
        """Display session information"""
        if not session_state:
            return
        
        session_start = datetime.fromisoformat(session_state.get("session_start"))
        elapsed = datetime.now() - session_start
        elapsed_minutes = int(elapsed.total_seconds() / 60)
        
        progress_percent = min(100, (elapsed_minutes / 240) * 100)
        
        print("┌─ SESSION INFORMATION ─────────────────────────────────────────────────────┐")
        print(f"│ Status:              {session_state.get('status', 'unknown').upper():50} │")
        print(f"│ Start Time:          {session_start.strftime('%Y-%m-%d %H:%M:%S'):50} │")
        print(f"│ Elapsed:             {self.format_duration(elapsed_minutes):50} │")
        print(f"│ Remaining:           {self.format_duration(240 - elapsed_minutes):50} │")
        print(f"│ Progress:            {progress_percent:.1f}% [{'█' * int(progress_percent/5)}{' ' * (20 - int(progress_percent/5))}] │")
        print("└────────────────────────────────────────────────────────────────────────────┘")
        print()
    
    def display_checkpoint_info(self, checkpoint_data):
        """Display latest checkpoint information"""
        if not checkpoint_data:
            print("⏳ No checkpoints yet...\n")
            return
        
        checkpoint_num = checkpoint_data.get("checkpoint_number", 0)
        elapsed_min = checkpoint_data.get("elapsed_minutes", 0)
        cycles = checkpoint_data.get("trading_cycles", 0)
        balance = checkpoint_data.get("balance_snapshot", {})
        
        print("┌─ LATEST CHECKPOINT ───────────────────────────────────────────────────────┐")
        print(f"│ Checkpoint #:        {checkpoint_num:50} │")
        print(f"│ At Elapsed Time:     {self.format_duration(elapsed_min):50} │")
        print(f"│ Trading Cycles:      {cycles:50} │")
        print("│ " + " " * 76 + "│")
        
        if balance:
            current = balance.get("current_balance", 0)
            peak = balance.get("peak_balance", 0)
            low = balance.get("lowest_balance", 0)
            change = balance.get("total_change", 0)
            return_pct = balance.get("return_percentage", 0)
            
            print(f"│ Current Balance:     ${current:,.2f} {' ' * (38 - len(f'{current:,.2f}'))}│")
            print(f"│ Peak Balance:        ${peak:,.2f} {' ' * (38 - len(f'{peak:,.2f}'))}│")
            print(f"│ Lowest Balance:      ${low:,.2f} {' ' * (38 - len(f'{low:,.2f}'))}│")
            print(f"│ Total Change:        ${change:,.2f} ({return_pct:+.2f}%) {' ' * (24 - len(f'{change:,.2f}'))}│")
        
        print("└────────────────────────────────────────────────────────────────────────────┘")
        print()
    
    def display_checkpoints_timeline(self):
        """Display all checkpoints timeline"""
        checkpoints = self.get_all_checkpoints()
        
        if not checkpoints:
            print("⏳ No checkpoints recorded yet\n")
            return
        
        print("┌─ CHECKPOINTS TIMELINE ────────────────────────────────────────────────────┐")
        
        for i, cp_file in enumerate(checkpoints, 1):
            try:
                with open(cp_file, "r") as f:
                    cp_data = json.load(f)
                
                cp_num = cp_data.get("checkpoint_number", i)
                elapsed_min = cp_data.get("elapsed_minutes", 0)
                cycles = cp_data.get("trading_cycles", 0)
                balance = cp_data.get("balance_snapshot", {})
                current = balance.get("current_balance", 0)
                
                status = "✅" if i < len(checkpoints) else "🔄"
                print(f"│ {status} CP#{cp_num:02d} @ {self.format_duration(elapsed_min):7} | {cycles:6} cycles | ${current:>10,.2f} │")
            except Exception as e:
                print(f"│ ⚠️  Error reading checkpoint: {str(e)[:60]:60} │")
        
        print("└────────────────────────────────────────────────────────────────────────────┘")
        print()
    
    def display_final_summary(self, final_state):
        """Display final session summary"""
        if not final_state:
            return
        
        print("┌─ FINAL SESSION SUMMARY ───────────────────────────────────────────────────┐")
        print(f"│ Status:              {final_state.get('status', 'unknown').upper():50} │")
        print(f"│ Total Duration:      {self.format_duration(final_state.get('total_duration_minutes', 0)):50} │")
        print(f"│ Total Cycles:        {final_state.get('total_cycles', 0):50} │")
        print(f"│ Total Checkpoints:   {final_state.get('total_checkpoints', 0):50} │")
        
        balance = final_state.get("balance_summary", {})
        if balance:
            print("│ " + " " * 76 + "│")
            initial = balance.get("initial_balance", 0)
            current = balance.get("current_balance", 0)
            peak = balance.get("peak_balance", 0)
            low = balance.get("lowest_balance", 0)
            total_change = balance.get("total_change", 0)
            return_pct = balance.get("return_percentage", 0)
            
            print(f"│ Initial Balance:     ${initial:,.2f} {' ' * (36 - len(f'{initial:,.2f}'))}│")
            print(f"│ Final Balance:       ${current:,.2f} {' ' * (36 - len(f'{current:,.2f}'))}│")
            print(f"│ Peak Balance:        ${peak:,.2f} {' ' * (36 - len(f'{peak:,.2f}'))}│")
            print(f"│ Lowest Balance:      ${low:,.2f} {' ' * (36 - len(f'{low:,.2f}'))}│")
            print(f"│ Total Change:        ${total_change:+,.2f} ({return_pct:+.2f}%) {' ' * (20 - len(f'{total_change:+,.2f}'))}│")
        
        print("└────────────────────────────────────────────────────────────────────────────┘")
        print()
    
    def display_commands(self):
        """Display helpful commands"""
        print("┌─ COMMANDS ────────────────────────────────────────────────────────────────┐")
        print("│ Monitor Log:         tail -f state/4hour_session.log                     │")
        print("│ View Checkpoint:     cat state/checkpoint_01.json | python3 -m json.tool │")
        print("│ Checkpoint Count:    ls -1 state/checkpoint_*.json | wc -l               │")
        print("│ Final Summary:       cat state/4hour_session_final.json | python3 -m json│")
        print("│                                                                          │")
        print("│ Press Ctrl+C to stop monitoring                                         │")
        print("└────────────────────────────────────────────────────────────────────────────┘")
        print()
    
    def run(self, continuous=True):
        """Run monitor dashboard"""
        try:
            while True:
                self.clear_screen()
                self.display_header()
                
                session_state = self.load_session_state()
                checkpoint_data = self.load_latest_checkpoint()
                final_state = self.load_final_state()
                
                if session_state:
                    self.display_session_info(session_state)
                    self.display_checkpoint_info(checkpoint_data)
                    self.display_checkpoints_timeline()
                    
                    if final_state:
                        self.display_final_summary(final_state)
                        print("✅ SESSION COMPLETED!")
                        print()
                        break
                else:
                    print("⏳ Waiting for session to start...\n")
                
                self.display_commands()
                
                if not continuous:
                    break
                
                time.sleep(self.refresh_interval)
        
        except KeyboardInterrupt:
            print("\n\n✅ Monitor stopped")
            sys.exit(0)


def main():
    """Main entry point"""
    monitor = FourHourSessionMonitor()
    monitor.run(continuous=True)


if __name__ == "__main__":
    main()
