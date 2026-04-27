#!/usr/bin/env python3
"""
Live Environment Integration - State Recovery & Auto-Recovery
Wraps all live trading systems with persistent memory
"""

import sys
from pathlib import Path
from auto_recovery import check_and_recover
from system_state_manager import SystemStateManager

# Initialize recovery on startup
def initialize_live_environment():
    """Initialize live environment with state recovery"""
    
    print("🔄 Initializing live environment with state recovery...")
    
    # 1. Check for restart recovery
    recovery_result = check_and_recover()
    if recovery_result['restart_detected']:
        print("✅ Recovered from restart")
        context = recovery_result['context']
        print(f"   Phase: {context['system_status']['current_phase']}")
        print(f"   Task: {context['system_status']['current_task']}")
        print(f"   Progress: {context['system_status']['progress']}")
    
    # 2. Initialize state manager
    state_mgr = SystemStateManager()
    
    # 3. Update to live phase
    state_mgr.update_operational_state(
        phase="live_trading",
        task="continuous_portfolio_management"
    )
    
    print("✅ Live environment initialized with state recovery")
    return state_mgr

# Decorator to track operations
def track_operation(operation_name: str):
    """Decorator to track operations in state manager"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            state_mgr = SystemStateManager()
            try:
                result = func(*args, **kwargs)
                state_mgr.record_task_completion(
                    task=operation_name,
                    status="success"
                )
                return result
            except Exception as e:
                state_mgr.record_error(
                    error=str(e),
                    context=operation_name
                )
                raise
        return wrapper
    return decorator

if __name__ == "__main__":
    state_mgr = initialize_live_environment()
    print("\n🚀 Live environment ready with state recovery")
