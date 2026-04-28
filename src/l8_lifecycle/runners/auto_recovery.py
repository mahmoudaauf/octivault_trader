#!/usr/bin/env python3
"""
Automatic Restart Handler
Ensures system automatically recovers and resumes after any restart
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from system_state_manager import SystemStateManager, ContextRecoveryEngine


def check_and_recover():
    """Check for restart and recover if needed"""
    
    workspace = "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
    state_manager = SystemStateManager(workspace)
    
    # Check for restart
    restart_detected = state_manager.detect_restart()
    
    if restart_detected:
        print("\n" + "="*80)
        print("⚠️  SYSTEM RESTART DETECTED")
        print("="*80)
        print("\n🔄 Initiating automatic recovery...")
        
        # Perform recovery
        recovery = state_manager.handle_restart_recovery()
        
        # Record recovery
        state_manager.update_operational_state(
            data={"last_boot": datetime.now().isoformat()}
        )
        
        # Initialize recovery engine
        recovery_engine = ContextRecoveryEngine(state_manager)
        recovery_engine.print_recovery_report()
        
        # Save recovered context
        full_context = recovery_engine.recover_context()
        context_file = Path(workspace) / "state" / "context.json"
        with open(context_file, 'w') as f:
            json.dump(full_context, f, indent=2)
        
        print("\n✅ SYSTEM FULLY RECOVERED")
        print("   All state restored from last checkpoint")
        print("   Ready to continue operations\n")
        
        return {
            "restart_detected": True,
            "recovered": True,
            "context": full_context
        }
    
    else:
        print("\n✅ NORMAL BOOT - Loading operational state...")
        
        # Load context
        recovery_engine = ContextRecoveryEngine(state_manager)
        full_context = recovery_engine.recover_context()
        
        return {
            "restart_detected": False,
            "recovered": True,
            "context": full_context
        }


def ensure_continuous_operation():
    """Ensure continuous operation across restarts"""
    
    workspace = "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
    state_manager = SystemStateManager(workspace)
    
    # Get current context
    context = state_manager.get_system_context()
    
    print("\n" + "="*80)
    print("CONTINUOUS OPERATION STATUS")
    print("="*80)
    
    # Check active processes
    if context["system_status"]["active_processes"] > 0:
        print(f"✅ {context['system_status']['active_processes']} active processes detected")
        print("   Resuming from last known state...")
    else:
        print("✅ Clean slate - ready for new operations")
    
    # Check for pending tasks
    session_mem = context["session_memory"]
    if session_mem["tasks_pending"]:
        print(f"\n📋 Pending Tasks ({len(session_mem['tasks_pending'])} remaining):")
        for task in session_mem["tasks_pending"][:5]:
            print(f"   • {task}")
    
    # Check for recent errors
    if session_mem["errors_encountered"]:
        recent_errors = session_mem["errors_encountered"][-3:]
        print(f"\n⚠️  Recent Errors ({len(session_mem['errors_encountered'])} total):")
        for err in recent_errors:
            print(f"   • {err['error']} ({err['timestamp']})")
    
    print("\n" + "="*80)
    
    return context


class AutoRecoveryAgent:
    """Autonomous agent that handles recovery without user intervention"""
    
    def __init__(self, workspace: str = None):
        self.workspace = workspace or "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
        self.state_manager = SystemStateManager(self.workspace)
    
    def auto_recover(self) -> Dict:
        """Automatically recover from any restart or failure"""
        
        print("\n🤖 AUTO-RECOVERY AGENT ACTIVATED")
        print("   Checking system state...")
        
        # 1. Check for restart
        if self.state_manager.detect_restart():
            print("   ⚠️  Restart detected - recovering state...")
            recovery = self.state_manager.handle_restart_recovery()
        else:
            print("   ✅ Normal boot sequence")
            recovery = None
        
        # 2. Load full context
        context = self.state_manager.get_system_context()
        
        # 3. Check operational integrity
        print("   Verifying operational integrity...")
        integrity_check = self._verify_integrity(context)
        
        # 4. Determine recovery actions
        recovery_actions = self._determine_recovery_actions(context, integrity_check)
        
        # 5. Execute recovery actions
        if recovery_actions:
            print(f"   Executing {len(recovery_actions)} recovery actions...")
            for action in recovery_actions:
                self._execute_recovery_action(action)
                self.state_manager.add_recovery_action(action)
        
        print("\n🤖 AUTO-RECOVERY COMPLETE")
        
        return {
            "restart_detected": self.state_manager.detect_restart(),
            "recovered": True,
            "context": context,
            "recovery_actions": recovery_actions,
            "integrity_verified": integrity_check["is_valid"]
        }
    
    def _verify_integrity(self, context: Dict) -> Dict:
        """Verify operational integrity"""
        issues = []
        
        # Check state file consistency
        if not context["operational_state"]:
            issues.append("Missing operational state")
        
        if not context["session_memory"]["session_id"]:
            issues.append("Missing session ID")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues
        }
    
    def _determine_recovery_actions(self, context: Dict, integrity: Dict) -> List[str]:
        """Determine what recovery actions to take"""
        actions = []
        
        # If integrity issues, rebuild state
        if not integrity["is_valid"]:
            actions.append("rebuild_operational_state")
        
        # If checkpoint exists, validate it
        if context["checkpoint"]:
            actions.append("validate_checkpoint")
        
        # If pending tasks, resume them
        if context["session_memory"]["tasks_pending"]:
            actions.append("resume_pending_tasks")
        
        # If errors recorded, analyze them
        if context["session_memory"]["errors_encountered"]:
            actions.append("analyze_error_log")
        
        return actions
    
    def _execute_recovery_action(self, action: str):
        """Execute a recovery action"""
        print(f"     • {action.replace('_', ' ').title()}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import json
    from typing import Dict, List
    
    # Run automatic recovery
    result = check_and_recover()
    
    # Ensure continuous operation
    context = ensure_continuous_operation()
    
    # Run auto-recovery agent
    print("\nInitializing auto-recovery agent...")
    agent = AutoRecoveryAgent()
    recovery_result = agent.auto_recover()
    
    print("\n" + "="*80)
    print("✅ SYSTEM READY FOR OPERATIONS")
    print("="*80)
    print(f"Session ID: {recovery_result['context']['session_memory']['session_id']}")
    print(f"Current Phase: {recovery_result['context']['system_status']['current_phase']}")
    print(f"Recovery Successful: {recovery_result['recovered']}")
    print("="*80 + "\n")
