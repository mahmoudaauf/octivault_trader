#!/usr/bin/env python3
"""Test script to diagnose TrendHunter import issues."""
import sys
import traceback

print("=" * 60)
print("TRENDHUNTER IMPORT DIAGNOSTIC")
print("=" * 60)

try:
    print("\n[1/3] Checking core.agent_registry...")
    from core import agent_registry
    print("✅ agent_registry imported")
    
    if "TrendHunter" in agent_registry.AGENT_CLASS_MAP:
        cls = agent_registry.AGENT_CLASS_MAP["TrendHunter"]
        print(f"✅ TrendHunter in AGENT_CLASS_MAP: {cls}")
        print(f"   Agent type: {getattr(cls, 'agent_type', 'UNKNOWN')}")
    else:
        print("❌ TrendHunter NOT in AGENT_CLASS_MAP")
        print(f"   Available agents: {list(agent_registry.AGENT_CLASS_MAP.keys())}")
    
    if "TrendHunter" in agent_registry.AGENT_IMPORT_ERRORS:
        error_info = agent_registry.AGENT_IMPORT_ERRORS["TrendHunter"]
        print(f"❌ TrendHunter import error detected:")
        print(f"   Error: {error_info.get('error')}")
        print(f"   Traceback:\n{error_info.get('traceback')}")
    else:
        print("✅ No import errors recorded for TrendHunter")

except Exception as e:
    print(f"❌ Failed to check agent_registry: {e}")
    traceback.print_exc()

print("\n[2/3] Direct TrendHunter import attempt...")
try:
    from agents.trend_hunter import TrendHunter
    print(f"✅ Direct import successful: {TrendHunter}")
    print(f"   Agent type: {getattr(TrendHunter, 'agent_type', 'UNKNOWN')}")
except Exception as e:
    print(f"❌ Direct import failed: {e}")
    traceback.print_exc()

print("\n[3/3] Checking agent_manager...")
try:
    from core.agent_manager import AgentManager
    print("✅ AgentManager imported")
except Exception as e:
    print(f"❌ AgentManager import failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
