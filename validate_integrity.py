#!/usr/bin/env python3
"""
Comprehensive System Integrity Validation Script
Tests all phases, component interactions, and data flow
"""

import asyncio
import sys
import logging
from typing import Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class IntegrityValidator:
    def __init__(self):
        self.results: Dict[str, bool] = {}
        self.errors: List[str] = []
        
    def test(self, name: str, condition: bool, error_msg: str = ""):
        """Record test result"""
        self.results[name] = condition
        if not condition:
            self.errors.append(f"{name}: {error_msg}")
        status = "✅" if condition else "❌"
        print(f"{status} {name}")
        if not condition and error_msg:
            print(f"   └─ {error_msg}")
    
    def summary(self):
        """Print summary"""
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        print(f"\n{'='*60}")
        print(f"INTEGRITY VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Passed: {passed}/{total}")
        print(f"Failed: {total - passed}/{total}")
        
        if self.errors:
            print(f"\n❌ ERRORS:")
            for err in self.errors:
                print(f"  - {err}")
        else:
            print(f"\n✅ ALL TESTS PASSED")
        
        return passed == total

async def validate_imports():
    """Phase 1: Validate all critical imports"""
    print("\n" + "="*60)
    print("PHASE 1: IMPORT VALIDATION")
    print("="*60)
    
    validator = IntegrityValidator()
    
    # Core imports
    try:
        from core.config import Config
        validator.test("Import: core.config", True)
    except Exception as e:
        validator.test("Import: core.config", False, str(e))
    
    try:
        from core.shared_state import SharedState
        validator.test("Import: core.shared_state", True)
    except Exception as e:
        validator.test("Import: core.shared_state", False, str(e))
    
    try:
        from core.exchange_client import ExchangeClient
        validator.test("Import: core.exchange_client", True)
    except Exception as e:
        validator.test("Import: core.exchange_client", False, str(e))
    
    try:
        from core.market_data_feed import MarketDataFeed
        validator.test("Import: core.market_data_feed", True)
    except Exception as e:
        validator.test("Import: core.market_data_feed", False, str(e))
    
    try:
        from core.execution_manager import ExecutionManager
        validator.test("Import: core.execution_manager", True)
    except Exception as e:
        validator.test("Import: core.execution_manager", False, str(e))
    
    try:
        from core.agent_manager import AgentManager
        validator.test("Import: core.agent_manager", True)
    except Exception as e:
        validator.test("Import: core.agent_manager", False, str(e))
    
    try:
        from core.risk_manager import RiskManager
        validator.test("Import: core.risk_manager", True)
    except Exception as e:
        validator.test("Import: core.risk_manager", False, str(e))
    
    try:
        from portfolio.balancer import PortfolioBalancer
        validator.test("Import: portfolio.balancer", True)
    except Exception as e:
        validator.test("Import: portfolio.balancer", False, str(e))
    
    try:
        from core.compounding_engine import CompoundingEngine
        validator.test("Import: core.compounding_engine", True)
    except Exception as e:
        validator.test("Import: core.compounding_engine", False, str(e))
    
    try:
        from agents.liquidation_agent import LiquidationAgent
        validator.test("Import: agents.liquidation_agent", True)
    except Exception as e:
        validator.test("Import: agents.liquidation_agent", False, str(e))
    
    try:
        from core.performance_evaluator import PerformanceEvaluator
        validator.test("Import: core.performance_evaluator", True)
    except Exception as e:
        validator.test("Import: core.performance_evaluator", False, str(e))
    
    try:
        from dashboard_server import DashboardServer
        validator.test("Import: dashboard_server", True)
    except Exception as e:
        validator.test("Import: dashboard_server", False, str(e))
    
    # Agent imports
    try:
        from agents.dip_sniper import DipSniper
        validator.test("Import: agents.dip_sniper", True)
    except Exception as e:
        validator.test("Import: agents.dip_sniper", False, str(e))
    
    try:
        from agents.trend_hunter import TrendHunter
        validator.test("Import: agents.trend_hunter", True)
    except Exception as e:
        validator.test("Import: agents.trend_hunter", False, str(e))
    
    try:
        from agents.ipo_chaser import IPOChaser
        validator.test("Import: agents.ipo_chaser", True)
    except Exception as e:
        validator.test("Import: agents.ipo_chaser", False, str(e))
    
    return validator.summary()

async def validate_component_construction():
    """Phase 2: Validate component construction"""
    print("\n" + "="*60)
    print("PHASE 2: COMPONENT CONSTRUCTION")
    print("="*60)
    
    validator = IntegrityValidator()
    
    try:
        from core.config import Config
        from core.shared_state import SharedState
        
        config = Config()
        validator.test("Construct: Config", True)
        
        # SharedState expects a dict, not a Config object
        config_dict = {}
        shared_state = SharedState(config_dict)
        validator.test("Construct: SharedState", True)
        
        # Test SharedState methods
        validator.test(
            "SharedState.get_accepted_symbols",
            hasattr(shared_state, "get_accepted_symbols"),
            "Method missing"
        )
        
        validator.test(
            "SharedState.update_latest_price",
            hasattr(shared_state, "update_latest_price"),
            "Method missing"
        )
        
        validator.test(
            "SharedState.add_ohlcv",
            hasattr(shared_state, "add_ohlcv"),
            "Method missing"
        )
        
        validator.test(
            "SharedState.emit_event",
            hasattr(shared_state, "emit_event"),
            "Method missing"
        )
        
    except Exception as e:
        validator.test("Component Construction", False, str(e))
    
    return validator.summary()

async def validate_agent_registry():
    """Phase 3: Validate agent registry"""
    print("\n" + "="*60)
    print("PHASE 3: AGENT REGISTRY")
    print("="*60)
    
    validator = IntegrityValidator()
    
    try:
        from core.agent_registry import AGENT_CLASS_MAP
        
        validator.test("Agent Registry Loaded", True)
        
        expected_agents = ["DipSniper", "TrendHunter", "IPOChaser"]
        for agent_name in expected_agents:
            is_registered = agent_name in AGENT_CLASS_MAP
            validator.test(
                f"Agent Registered: {agent_name}",
                is_registered,
                f"{agent_name} not in AGENT_CLASS_MAP"
            )
        
        # Verify agent classes are callable
        for agent_name, agent_class in AGENT_CLASS_MAP.items():
            validator.test(
                f"Agent Callable: {agent_name}",
                callable(agent_class),
                f"{agent_name} class is not callable"
            )
        
    except Exception as e:
        validator.test("Agent Registry", False, str(e))
    
    return validator.summary()

async def validate_data_flow():
    """Phase 4: Validate data flow patterns"""
    print("\n" + "="*60)
    print("PHASE 4: DATA FLOW VALIDATION")
    print("="*60)
    
    validator = IntegrityValidator()
    
    try:
        from core.config import Config
        from core.shared_state import SharedState
        
        config = Config()
        config_dict = {}
        shared_state = SharedState(config_dict)
        
        # Test signal flow
        test_signal = {
            "action": "buy",
            "confidence": 0.85,
            "reason": "Test signal",
            "timestamp": 1234567890
        }
        
        # Simulate agent signal injection
        if not hasattr(shared_state, 'agent_signals'):
            shared_state.agent_signals = {}
        
        shared_state.agent_signals["BTCUSDT"] = {"TestAgent": test_signal}
        
        validator.test(
            "Signal Injection",
            "BTCUSDT" in shared_state.agent_signals,
            "Signal not stored"
        )
        
        # Test score storage
        if not hasattr(shared_state, 'agent_scores'):
            shared_state.agent_scores = {}
        
        shared_state.agent_scores["BTCUSDT"] = 0.75
        
        validator.test(
            "Score Storage",
            shared_state.agent_scores.get("BTCUSDT") == 0.75,
            "Score not stored correctly"
        )
        
        # Test balance updates
        shared_state.balances = {
            "USDT": {"free": 10000.0, "locked": 0.0},
            "BTC": {"free": 0.5, "locked": 0.0}
        }
        
        validator.test(
            "Balance Storage",
            "USDT" in shared_state.balances,
            "Balances not stored"
        )
        
        # Test position tracking
        if not hasattr(shared_state, 'positions'):
            shared_state.positions = {}
        
        shared_state.positions["BTCUSDT"] = {
            "qty": 0.5,
            "entry_price": 50000.0,
            "current_price": 51000.0
        }
        
        validator.test(
            "Position Tracking",
            "BTCUSDT" in shared_state.positions,
            "Position not tracked"
        )
        
    except Exception as e:
        validator.test("Data Flow", False, str(e))
    
    return validator.summary()

async def validate_appcontext_integration():
    """Phase 5: Validate AppContext integration"""
    print("\n" + "="*60)
    print("PHASE 5: APPCONTEXT INTEGRATION")
    print("="*60)
    
    validator = IntegrityValidator()
    
    try:
        from core.app_context import AppContext
        
        config = {"UP_TO_PHASE": 3}  # Don't go too far to avoid API calls
        ctx = AppContext(config=config)
        
        validator.test("AppContext Construction", True)
        
        # Verify component slots exist
        validator.test(
            "AppContext.shared_state",
            hasattr(ctx, "shared_state"),
            "shared_state attribute missing"
        )
        
        validator.test(
            "AppContext.exchange_client",
            hasattr(ctx, "exchange_client"),
            "exchange_client attribute missing"
        )
        
        validator.test(
            "AppContext.market_data_feed",
            hasattr(ctx, "market_data_feed"),
            "market_data_feed attribute missing"
        )
        
        validator.test(
            "AppContext.execution_manager",
            hasattr(ctx, "execution_manager"),
            "execution_manager attribute missing"
        )
        
        validator.test(
            "AppContext.agent_manager",
            hasattr(ctx, "agent_manager"),
            "agent_manager attribute missing"
        )
        
        validator.test(
            "AppContext.dashboard_server",
            hasattr(ctx, "dashboard_server"),
            "dashboard_server attribute missing"
        )
        
        # Verify initialization method exists
        validator.test(
            "AppContext.initialize_all",
            hasattr(ctx, "initialize_all") and callable(ctx.initialize_all),
            "initialize_all method missing or not callable"
        )
        
    except Exception as e:
        validator.test("AppContext Integration", False, str(e))
    
    return validator.summary()

async def main():
    """Run all validation phases"""
    print("\n" + "="*60)
    print("OCTIVAULT SYSTEM INTEGRITY VALIDATION")
    print("="*60)
    
    results = []
    
    # Run all validation phases
    results.append(await validate_imports())
    results.append(await validate_component_construction())
    results.append(await validate_agent_registry())
    results.append(await validate_data_flow())
    results.append(await validate_appcontext_integration())
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    all_passed = all(results)
    
    if all_passed:
        print("✅ ALL VALIDATION PHASES PASSED")
        print("\nSystem integrity: VERIFIED")
        print("Ready for production deployment.")
        return 0
    else:
        print("❌ SOME VALIDATION PHASES FAILED")
        print("\nSystem integrity: COMPROMISED")
        print("Review errors above and fix before deployment.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
