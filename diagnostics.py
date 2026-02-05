# diagnostics.py

import asyncio
import logging
from datetime import datetime

from core.config import Config
from core.app_context import AppContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Diagnostics")


async def run_diagnostics(app_context):
    print("\n📋 ====== Octivault Trader Diagnostics Report ======\n")

    # 1. Try to query health updates via method (if available)
    print("🩺 Component Health Report (via update_system_health or timestamps):")

    if hasattr(app_context, 'shared_state') and app_context.shared_state:
        shared_state = app_context.shared_state

        # Check last heartbeat timestamps if available
        if hasattr(shared_state, 'timestamps'):
            now = datetime.utcnow().timestamp()
            print("\n⏱  Last Heartbeats:")
            for component, ts in shared_state.timestamps.items():
                age = now - ts
                status = "🟢" if age < 30 else "🟡" if age < 90 else "🔴"
                print(f"   - {component:20s}: {status} {age:.1f}s ago")
        else:
            print("⚠️ shared_state.timestamps not available.")
    else:
        print("❌ SharedState is missing in app_context.")

    # 2. Agent registration check
    print("\n🤖 Registered Agents:")
    if hasattr(app_context, 'agents') and isinstance(app_context.agents, dict):
        if not app_context.agents:
            print("   ⚠️ No agents registered.")
        for name, agent in app_context.agents.items():
            print(f"   - {name:20s}: class={agent.__class__.__name__}")
    else:
        print("❌ app_context.agents is missing or invalid.")

    # 3. AgentManager listing
    print("\n📅 AgentManager Schedule:")
    if hasattr(app_context, 'agent_manager'):
        am = app_context.agent_manager
        if hasattr(am, 'agents') and am.agents:
            for name, agent in am.agents.items():
                print(f"   - {name:20s} in AgentManager")
        else:
            print("   ⚠️ AgentManager has no agents registered.")
    else:
        print("❌ AgentManager not found in app_context.")

    print("\n✅ Diagnostics complete.\n")


async def main():
    config = Config()
    app_context = AppContext(config)
    await app_context.initialize_all(up_to_phase=9)
    await run_diagnostics(app_context)


if __name__ == "__main__":
    asyncio.run(main())
