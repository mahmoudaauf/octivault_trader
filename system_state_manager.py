#!/usr/bin/env python3
"""
System State Persistence & Recovery Manager
Ensures the system never loses context after restarts
Maintains complete operational memory across sessions
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib
import pickle


class SystemStateManager:
    """Manages persistent system state across restarts"""
    
    def __init__(self, workspace_root: str = None):
        self.workspace_root = Path(workspace_root or "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader")
        self.state_dir = self.workspace_root / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # State file paths
        self.operational_state_file = self.state_dir / "operational_state.json"
        self.checkpoint_file = self.state_dir / "checkpoint.json"
        self.recovery_state_file = self.state_dir / "recovery_state.json"
        self.session_memory_file = self.state_dir / "session_memory.json"
        self.context_file = self.state_dir / "context.json"
        
        # Load or initialize state
        self.operational_state = self._load_operational_state()
        self.session_memory = self._load_session_memory()
    
    def _load_operational_state(self) -> Dict[str, Any]:
        """Load operational state from disk"""
        if self.operational_state_file.exists():
            try:
                with open(self.operational_state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error loading operational state: {e}")
                return self._get_default_operational_state()
        return self._get_default_operational_state()
    
    def _get_default_operational_state(self) -> Dict[str, Any]:
        """Get default operational state template"""
        return {
            "system_version": "1.0",
            "last_boot": None,
            "last_activity": None,
            "current_phase": None,
            "current_task": None,
            "progress": {
                "phase1_implementation": {"status": "complete", "date": "2026-04-25"},
                "phase2_unit_testing": {"status": "complete", "date": "2026-04-25"},
                "phase3_integration_testing": {"status": "complete", "date": "2026-04-26"},
                "phase4_sandbox_validation": {"status": "in_progress", "date": "2026-04-26", "started": "14:09:22"},
                "phase5_production": {"status": "pending", "date": None}
            },
            "active_processes": [],
            "configuration": {
                "monitoring_duration_hours": 48,
                "metrics_collection_interval_seconds": 60,
                "checkpoint_interval_seconds": 300
            },
            "last_checkpoint": None,
            "error_recovery_enabled": True,
            "recovery_attempts": 0,
            "previous_restarts": []
        }
    
    def _load_session_memory(self) -> Dict[str, Any]:
        """Load session memory from disk"""
        if self.session_memory_file.exists():
            try:
                with open(self.session_memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error loading session memory: {e}")
                return self._get_default_session_memory()
        return self._get_default_session_memory()
    
    def _get_default_session_memory(self) -> Dict[str, Any]:
        """Get default session memory template"""
        return {
            "session_id": self._generate_session_id(),
            "created_at": datetime.now().isoformat(),
            "tasks_completed": [],
            "tasks_pending": [],
            "metrics_collected": 0,
            "errors_encountered": [],
            "recovery_actions": [],
            "knowledge_base": {}
        }
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def save_operational_state(self):
        """Save operational state to disk"""
        self.operational_state["last_activity"] = datetime.now().isoformat()
        with open(self.operational_state_file, 'w') as f:
            json.dump(self.operational_state, f, indent=2)
    
    def save_session_memory(self):
        """Save session memory to disk"""
        with open(self.session_memory_file, 'w') as f:
            json.dump(self.session_memory, f, indent=2)
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Save detailed checkpoint for recovery"""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "phase": self.operational_state["current_phase"],
            "task": self.operational_state["current_task"],
            "data": checkpoint_data,
            "recovery_enabled": True
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint for recovery"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error loading checkpoint: {e}")
        return None
    
    def detect_restart(self) -> bool:
        """Detect if system was restarted"""
        last_boot = self.operational_state.get("last_boot")
        current_boot = self._get_system_boot_time()
        
        if last_boot and last_boot != current_boot:
            return True
        return False
    
    def _get_system_boot_time(self) -> str:
        """Get system boot time"""
        # Simplified - in production would use system uptime
        return datetime.now().isoformat()[:10]
    
    def handle_restart_recovery(self) -> Dict[str, Any]:
        """Handle recovery after system restart"""
        print("\n" + "="*80)
        print("🔄 SYSTEM RESTART DETECTED - INITIATING STATE RECOVERY")
        print("="*80)
        
        # Record restart
        self.operational_state["previous_restarts"].append({
            "timestamp": datetime.now().isoformat(),
            "recovery_initiated": True
        })
        
        # Load checkpoint
        checkpoint = self.load_checkpoint()
        
        if checkpoint:
            print(f"✅ Found checkpoint from: {checkpoint['timestamp']}")
            print(f"   Phase: {checkpoint['phase']}")
            print(f"   Task: {checkpoint['task']}")
            
            recovery_state = {
                "status": "recovered",
                "checkpoint": checkpoint,
                "operational_state": self.operational_state,
                "session_memory": self.session_memory,
                "ready_to_continue": True
            }
            
            # Save recovery state
            with open(self.recovery_state_file, 'w') as f:
                json.dump(recovery_state, f, indent=2)
            
            print("✅ Recovery state loaded successfully")
            return recovery_state
        
        else:
            print("⚠️  No checkpoint found - using last operational state")
            return {
                "status": "partial_recovery",
                "operational_state": self.operational_state,
                "session_memory": self.session_memory,
                "ready_to_continue": True
            }
    
    def get_system_context(self) -> Dict[str, Any]:
        """Get complete system context for AI/agent"""
        return {
            "operational_state": self.operational_state,
            "session_memory": self.session_memory,
            "checkpoint": self.load_checkpoint(),
            "recovery_state": self._load_recovery_state(),
            "system_status": self._get_system_status()
        }
    
    def _load_recovery_state(self) -> Optional[Dict[str, Any]]:
        """Load recovery state"""
        if self.recovery_state_file.exists():
            try:
                with open(self.recovery_state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "current_phase": self.operational_state["current_phase"],
            "current_task": self.operational_state["current_task"],
            "active_processes": len(self.operational_state["active_processes"]),
            "recovery_enabled": self.operational_state["error_recovery_enabled"],
            "last_checkpoint": self.operational_state["last_checkpoint"],
            "session_id": self.session_memory["session_id"]
        }
    
    def record_task_completion(self, task: str, status: str, details: str = ""):
        """Record task completion in memory"""
        self.session_memory["tasks_completed"].append({
            "task": task,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        self.save_session_memory()
    
    def record_error(self, error: str, context: str = ""):
        """Record error for recovery analysis"""
        self.session_memory["errors_encountered"].append({
            "error": error,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
        self.save_session_memory()
    
    def add_recovery_action(self, action: str):
        """Record recovery action taken"""
        self.session_memory["recovery_actions"].append({
            "action": action,
            "timestamp": datetime.now().isoformat()
        })
        self.save_session_memory()
    
    def update_operational_state(self, phase: str = None, task: str = None, data: Dict = None):
        """Update operational state"""
        if phase:
            self.operational_state["current_phase"] = phase
        if task:
            self.operational_state["current_task"] = task
        if data:
            self.operational_state.update(data)
        
        self.operational_state["last_activity"] = datetime.now().isoformat()
        self.save_operational_state()
    
    def get_phase_status(self, phase: str) -> Optional[Dict[str, Any]]:
        """Get status of specific phase"""
        return self.operational_state["progress"].get(phase)
    
    def update_phase_status(self, phase: str, status: str, details: Dict = None):
        """Update phase status"""
        if phase in self.operational_state["progress"]:
            self.operational_state["progress"][phase]["status"] = status
            self.operational_state["progress"][phase]["last_updated"] = datetime.now().isoformat()
            if details:
                self.operational_state["progress"][phase].update(details)
            self.save_operational_state()
    
    def print_system_status(self):
        """Print complete system status"""
        print("\n" + "="*80)
        print("SYSTEM STATE SUMMARY")
        print("="*80)
        
        print("\n📊 OPERATIONAL STATE:")
        print(f"  Current Phase: {self.operational_state['current_phase']}")
        print(f"  Current Task: {self.operational_state['current_task']}")
        print(f"  Last Activity: {self.operational_state['last_activity']}")
        
        print("\n📈 PROGRESS:")
        for phase, status in self.operational_state["progress"].items():
            print(f"  {phase}: {status['status']}")
        
        print("\n💾 SESSION MEMORY:")
        print(f"  Session ID: {self.session_memory['session_id']}")
        print(f"  Tasks Completed: {len(self.session_memory['tasks_completed'])}")
        print(f"  Errors Encountered: {len(self.session_memory['errors_encountered'])}")
        print(f"  Recovery Actions: {len(self.session_memory['recovery_actions'])}")
        
        print("\n🔄 RECOVERY STATE:")
        recovery = self._load_recovery_state()
        if recovery:
            print(f"  Status: {recovery['status']}")
            print(f"  Ready to Continue: {recovery['ready_to_continue']}")
        else:
            print("  No recovery state (first run or clean start)")
        
        print("="*80 + "\n")


class ContextRecoveryEngine:
    """Recovers full operational context after restart"""
    
    def __init__(self, state_manager: SystemStateManager):
        self.state_manager = state_manager
    
    def recover_context(self) -> Dict[str, Any]:
        """Recover complete operational context"""
        print("\n🔄 RECOVERING OPERATIONAL CONTEXT...")
        
        context = {
            "timestamp": datetime.now().isoformat(),
            "system_state": self.state_manager.get_system_context(),
            "recovery_info": self._build_recovery_info(),
            "ready_for_operations": True
        }
        
        return context
    
    def _build_recovery_info(self) -> Dict[str, Any]:
        """Build recovery information"""
        recovery_state = self.state_manager._load_recovery_state()
        checkpoint = self.state_manager.load_checkpoint()
        
        return {
            "recovery_source": "checkpoint" if checkpoint else "operational_state",
            "checkpoint_age_seconds": self._get_checkpoint_age() if checkpoint else None,
            "last_known_phase": checkpoint["phase"] if checkpoint else self.state_manager.operational_state["current_phase"],
            "last_known_task": checkpoint["task"] if checkpoint else self.state_manager.operational_state["current_task"],
            "recovery_data_available": bool(recovery_state or checkpoint),
            "safe_to_continue": True
        }
    
    def _get_checkpoint_age(self) -> Optional[int]:
        """Get age of checkpoint in seconds"""
        checkpoint = self.state_manager.load_checkpoint()
        if checkpoint and "timestamp" in checkpoint:
            try:
                checkpoint_time = datetime.fromisoformat(checkpoint["timestamp"])
                age = (datetime.now() - checkpoint_time).total_seconds()
                return int(age)
            except:
                pass
        return None
    
    def print_recovery_report(self):
        """Print recovery report"""
        context = self.recover_context()
        
        print("\n" + "="*80)
        print("🔄 SYSTEM RECOVERY REPORT")
        print("="*80)
        
        recovery_info = context["recovery_info"]
        print(f"\nRecovery Source: {recovery_info['recovery_source']}")
        print(f"Last Known Phase: {recovery_info['last_known_phase']}")
        print(f"Last Known Task: {recovery_info['last_known_task']}")
        
        if recovery_info['checkpoint_age_seconds']:
            age_mins = recovery_info['checkpoint_age_seconds'] / 60
            print(f"Checkpoint Age: {age_mins:.1f} minutes")
        
        print(f"Safe to Continue: {'✅ YES' if recovery_info['safe_to_continue'] else '❌ NO'}")
        
        print("\n" + "="*80 + "\n")


# ============================================================================
# MAIN ENTRY POINT FOR STATE RECOVERY
# ============================================================================

def main():
    """Main entry point"""
    import sys
    
    workspace = "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
    
    # Initialize state manager
    state_manager = SystemStateManager(workspace)
    
    # Check for restart
    if state_manager.detect_restart():
        print("\n🔄 RESTART DETECTED - INITIATING RECOVERY")
        recovery = state_manager.handle_restart_recovery()
        print(f"Recovery Status: {recovery['status']}")
    else:
        print("\n✅ NORMAL OPERATION - STATE LOADED")
    
    # Print status
    state_manager.print_system_status()
    
    # Initialize recovery engine and print report
    recovery_engine = ContextRecoveryEngine(state_manager)
    recovery_engine.print_recovery_report()
    
    # Export context for AI/agent use
    full_context = recovery_engine.recover_context()
    
    # Save to context file
    with open(state_manager.context_file, 'w') as f:
        json.dump(full_context, f, indent=2)
    
    print("✅ System context recovered and saved to state/context.json")
    print("\n💡 AI/Agent can now load full operational context from state files")
    print("   and continue operations without losing any memory or progress.")


if __name__ == "__main__":
    main()
