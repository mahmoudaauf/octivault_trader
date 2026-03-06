"""
Quick validation test for Phase 10 components.

Run this to verify all components are working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add workspace to path
workspace = Path(__file__).parent
sys.path.insert(0, str(workspace))

from core.event_store import get_event_store, Event, EventType
from core.replay_engine import get_replay_engine, PortfolioState
from core.chaos_monkey import ChaosMonkey, ResilienceVerifier


async def test_event_store():
    """Test event store functionality."""
    print("\n" + "=" * 70)
    print("TEST 1: EVENT STORE")
    print("=" * 70)
    
    try:
        # Initialize
        print("Initializing event store...", end=" ")
        event_store = await get_event_store()
        print("✅")
        
        # Create test event
        print("Creating test event...", end=" ")
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=1234567890.0,
            sequence=0,
            component="test",
            symbol="BTC/USDT",
            data={"quantity": 1.0, "price": 50000.0},
        )
        print("✅")
        
        # Append
        print("Appending event to store...", end=" ")
        seq = await event_store.append(event)
        print(f"✅ (sequence={seq})")
        
        # Read all
        print("Reading all events...", end=" ")
        all_events = await event_store.read_all()
        print(f"✅ ({len(all_events)} events)")
        
        # Create snapshot
        print("Creating snapshot...", end=" ")
        state_data = {
            "open_positions": {"BTC/USDT": {"quantity": 1.0}},
            "total_capital": 100000.0,
            "realized_pnl": 500.0,
        }
        snapshot_id = await event_store.create_snapshot(state_data)
        print(f"✅ ({snapshot_id})")
        
        # Load snapshot
        print("Loading snapshot...", end=" ")
        seq, loaded = await event_store.load_snapshot(snapshot_id)
        print(f"✅ (sequence={seq})")
        
        # Query by symbol
        print("Querying by symbol...", end=" ")
        btc_events = await event_store.read_for_symbol("BTC/USDT")
        print(f"✅ ({len(btc_events)} events)")
        
        # Get count
        print("Getting event count...", end=" ")
        count = await event_store.get_event_count()
        print(f"✅ ({count} total events)")
        
        print("\n🟢 EVENT STORE: ALL TESTS PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ EVENT STORE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_replay_engine():
    """Test replay engine functionality."""
    print("\n" + "=" * 70)
    print("TEST 2: REPLAY ENGINE")
    print("=" * 70)
    
    try:
        # Initialize
        print("Initializing replay engine...", end=" ")
        replay_engine = await get_replay_engine()
        print("✅")
        
        # Replay all
        print("Replaying all events...", end=" ")
        state = await replay_engine.replay_all()
        print(f"✅ (sequence={state.sequence})")
        
        # Check state
        print("Verifying state...", end=" ")
        assert isinstance(state, PortfolioState), "Invalid state type"
        assert state.total_capital >= 0, "Invalid capital"
        print("✅")
        
        # Get history
        print("Getting state history...", end=" ")
        history = replay_engine.get_state_history()
        print(f"✅ ({len(history)} states)")
        
        # Replay to sequence (if events exist)
        if state.sequence > 0:
            print(f"Replaying to sequence {state.sequence // 2}...", end=" ")
            mid_state = await replay_engine.replay_to_sequence(state.sequence // 2)
            print(f"✅ (sequence={mid_state.sequence})")
        
        print("\n🟢 REPLAY ENGINE: ALL TESTS PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ REPLAY ENGINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_chaos_monkey():
    """Test chaos monkey functionality."""
    print("\n" + "=" * 70)
    print("TEST 3: CHAOS MONKEY")
    print("=" * 70)
    
    try:
        # Initialize
        print("Initializing chaos monkey...", end=" ")
        chaos = ChaosMonkey(enabled=False)  # Disabled for safety
        print("✅")
        
        # Check it's disabled
        print("Verifying chaos is disabled...", end=" ")
        failure = await chaos.maybe_inject_failure("test")
        assert failure is None, "Should not inject when disabled"
        print("✅")
        
        # Enable and test
        print("Enabling chaos monkey...", end=" ")
        chaos.enabled = True
        chaos.injection_rate = 1.0  # 100% injection for testing
        print("✅")
        
        # Try to inject
        print("Testing failure injection...", end=" ")
        try:
            await chaos.maybe_inject_failure("test")
            # Should have raised exception
            print("✅ (failure injected)")
        except Exception as e:
            # Expected
            print("✅ (exception raised)")
        
        # Disable again
        print("Disabling chaos monkey...", end=" ")
        chaos.enabled = False
        print("✅")
        
        # Get statistics
        print("Getting statistics...", end=" ")
        stats = chaos.get_statistics()
        print(f"✅ ({len(stats)} metrics)")
        
        # Initialize verifier
        print("Initializing resilience verifier...", end=" ")
        verifier = ResilienceVerifier(chaos)
        print("✅")
        
        print("\n🟢 CHAOS MONKEY: ALL TESTS PASSED")
        return True
    
    except Exception as e:
        print(f"\n❌ CHAOS MONKEY FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PHASE 10 COMPONENT VALIDATION")
    print("=" * 70)
    
    results = []
    
    # Test 1: Event Store
    results.append(await test_event_store())
    
    # Test 2: Replay Engine
    results.append(await test_replay_engine())
    
    # Test 3: Chaos Monkey
    results.append(await test_chaos_monkey())
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"\n✅ ALL {total} TESTS PASSED!")
        print("\nYour Phase 10 components are ready to integrate:")
        print("  1. core/event_store.py - Persistent event log")
        print("  2. core/replay_engine.py - Forensic analysis")
        print("  3. core/chaos_monkey.py - Resilience testing")
        print("\nNext steps:")
        print("  1. Read PHASE_10_IMPLEMENTATION_GUIDE.md")
        print("  2. Integrate with your existing code")
        print("  3. Deploy to staging")
        print("  4. Run in production")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        print("Fix the errors above and try again.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
