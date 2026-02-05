
import asyncio
import time
from core.stubs import is_fresh, MetaPolicy, KernelState
from core.agent_optimizer import load_tuned_params
from core.model_trainer import ModelTrainer
from types import SimpleNamespace

async def test_stubs():
    print("Testing Stubs...")
    
    # Test is_fresh
    print("- Testing is_fresh...", end="")
    shared_state = SimpleNamespace(_last_tick_timestamps={"BTCUSDT": time.time()})
    assert await is_fresh(shared_state, "BTCUSDT", max_age_sec=10) == True
    # Test stale data
    shared_state_stale = SimpleNamespace(_last_tick_timestamps={"BTCUSDT": time.time() - 100})
    assert await is_fresh(shared_state_stale, "BTCUSDT", max_age_sec=10) == False
    print(" OK")

    # Test MetaPolicy
    print("- Testing MetaPolicy...", end="")
    pol = MetaPolicy(state=KernelState(), min_conf=0.7)
    assert pol.evaluate({"confidence": 0.8}) == True
    assert pol.evaluate({"confidence": 0.6}) == False
    print(" OK")
    
    # Test Agent Optimizer
    print("- Testing load_tuned_params...", end="")
    params = load_tuned_params("NonExistentAgent")
    assert isinstance(params, dict)
    print(" OK")
    
    # Test Model Trainer
    print("- Testing ModelTrainer...", end="")
    trainer = ModelTrainer()
    assert trainer.train() == True
    print(" OK")

    print("\nAll Stub Tests Passed!")

if __name__ == "__main__":
    asyncio.run(test_stubs())
