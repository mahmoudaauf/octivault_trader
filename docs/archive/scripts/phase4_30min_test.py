#!/usr/bin/env python3
"""
Phase 4: 30-Minute Test Run
Quick validation before applying to live environment
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from monitoring.sandbox_monitor import SandboxMonitor
from system_state_manager import SystemStateManager
from auto_recovery import check_and_recover

async def run_30min_test():
    """Run 30-minute Phase 4 test with state recovery"""
    
    print("\n" + "="*80)
    print("PHASE 4: 30-MINUTE TEST RUN")
    print("Validation before applying to live environment")
    print("="*80 + "\n")
    
    # 1. Check for any restart recovery needed
    print("📋 Checking for restart recovery...")
    recovery_result = check_and_recover()
    if recovery_result['restart_detected']:
        print(f"✅ Recovered from previous restart")
        print(f"   Context: {recovery_result['context']['system_status']['current_phase']}")
    else:
        print(f"✅ Fresh start - no restart detected")
    
    # 2. Initialize state manager
    print("\n📊 Initializing state management...")
    state_mgr = SystemStateManager()
    state_mgr.update_operational_state(
        phase="phase4_test",
        task="30_minute_validation_test"
    )
    print(f"✅ State manager initialized")
    
    # 3. Start 30-minute monitoring (0.5 hours)
    print("\n🔄 Starting 30-minute monitoring test...")
    print("   Duration: 30 minutes (0.5 hours)")
    print("   Cycle interval: 60 seconds")
    print("   Expected cycles: 30")
    print("   Start time:", datetime.now().isoformat())
    
    monitor = SandboxMonitor()
    
    try:
        # Run for 0.5 hours (30 minutes)
        await monitor.start_monitoring(duration_hours=0.5)
        
        # 4. Save checkpoint
        print("\n💾 Saving test checkpoint...")
        state_mgr.save_checkpoint({
            "test_duration_minutes": 30,
            "cycles_completed": monitor.cycle_count,
            "metrics_collected": len(monitor.metrics_history),
            "consolidations": monitor.consolidation_events,
            "errors": monitor.total_errors,
            "status": "passed"
        })
        
        # 5. Generate results
        print("\n📈 TEST RESULTS:")
        print("-" * 80)
        print(f"Cycles completed:        {monitor.cycle_count}")
        print(f"Metrics collected:       {len(monitor.metrics_history)}")
        print(f"Health transitions:      {len(monitor.health_transitions)}")
        print(f"Consolidation events:    {monitor.consolidation_events}")
        print(f"Total errors:            {monitor.total_errors}")
        print(f"Test status:             {'✅ PASSED' if monitor.total_errors == 0 else '❌ FAILED'}")
        
        # 6. Record completion
        state_mgr.record_task_completion(
            task="phase4_30min_test",
            status="success",
            context={
                "cycles": monitor.cycle_count,
                "metrics": len(monitor.metrics_history),
                "errors": monitor.total_errors
            }
        )
        
        print("\n" + "="*80)
        print("✅ 30-MINUTE TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\n🚀 Next Step: Apply state recovery to live environment")
        print("   Run: python3 apply_recovery_to_live.py\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        state_mgr.record_error(
            error=str(e),
            context="30_minute_test_run"
        )
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_30min_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
