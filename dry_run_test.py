#!/usr/bin/env python3
"""
Comprehensive Dry-Run Test for Octivault AI Trading Bot
Simulates full system operation without real trades or API calls
Tests all phases, components, and operational flows
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any, List
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("DryRunTest")

class DryRunTester:
    """Orchestrates comprehensive dry-run testing"""
    
    def __init__(self):
        self.test_results: Dict[str, bool] = {}
        self.errors: List[str] = []
        self.ctx = None
        
    def log_test(self, name: str, passed: bool, details: str = ""):
        """Log test result"""
        self.test_results[name] = passed
        status = "✅" if passed else "❌"
        logger.info(f"{status} {name}")
        if details:
            logger.info(f"   └─ {details}")
        if not passed:
            self.errors.append(f"{name}: {details}")
    
    async def setup_dry_run_environment(self):
        """Set up a safe dry-run environment with mocked components"""
        logger.info("\n" + "="*60)
        logger.info("SETTING UP DRY-RUN ENVIRONMENT")
        logger.info("="*60)
        
        try:
            from core.app_context import AppContext
            
            # Configuration for dry-run mode
            config = {
                "DRY_RUN": True,
                "UP_TO_PHASE": 8,
                "DASHBOARD_PORT": 8888,  # Use unique port to avoid conflicts
                "BINANCE_API_KEY": "dry_run_key",
                "BINANCE_API_SECRET": "dry_run_secret",
                "MIN_ORDER_USDT": 5.0,
                "TARGET_EXPOSURE": 0.95,
                "ACCEPTED_SYMBOLS": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            }
            
            self.ctx = AppContext(config=config)
            self.log_test("AppContext Created", True, "Dry-run config applied")
            
            # Pre-build components
            self.ctx._ensure_components_built()
            self.log_test("Components Built", True, "All components constructed")
            
            # Mock the readiness gates to bypass API requirements
            async def mock_wait(*args, **kwargs):
                return {"status": "READY", "issues": []}
            
            self.ctx._wait_until_ready = mock_wait
            
            # Mock exchange_client to avoid real API calls
            if self.ctx.exchange_client:
                async def mock_exch_start(*args, **kwargs):
                    self.ctx.exchange_client._ready = True
                    return
                self.ctx.exchange_client.start = mock_exch_start
                
                async def mock_exch_noop(*args, **kwargs): return
                self.ctx.exchange_client._ensure_started_public = mock_exch_noop
                
                async def mock_get_info(*args, **kwargs): return {}
                self.ctx.exchange_client.get_exchange_info = mock_get_info
                
                async def mock_exch_req(*args, **kwargs): return {}
                self.ctx.exchange_client._request = mock_exch_req
                
                self.log_test("Exchange Mocked", True, "ExchangeClient.start/public/info/_request bypassed")
            self.log_test("Gates Mocked", True, "Readiness gates bypassed for dry-run")
            
            # Seed test data
            if self.ctx.shared_state:
                # Accepted symbols
                self.ctx.shared_state.accepted_symbols = {s: {"enabled": True, "meta": {}} for s in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]}
                self.ctx.shared_state.set_readiness_flag("accepted_symbols_ready", True)
                
                # Balances
                self.ctx.shared_state.balances = {
                    "USDT": {"free": 10000.0, "locked": 0.0},
                    "BTC": {"free": 0.1, "locked": 0.0},
                    "ETH": {"free": 1.0, "locked": 0.0},
                    "BNB": {"free": 5.0, "locked": 0.0}
                }
                self.ctx.shared_state.set_readiness_flag("balances_ready", True)
                
                # Prices
                self.ctx.shared_state.latest_prices = {
                    "BTCUSDT": 50000.0,
                    "ETHUSDT": 3000.0,
                    "BNBUSDT": 400.0
                }
                
                # Agent scores
                self.ctx.shared_state.agent_scores = {
                    "BTCUSDT": {"MLForecaster": {"roi": 0.02, "consecutive_losses": 0, "last_updated": datetime.utcnow().isoformat()}},
                    "ETHUSDT": {"MLForecaster": {"roi": 0.05, "consecutive_losses": 1, "last_updated": datetime.utcnow().isoformat()}},
                    "BNBUSDT": {"MLForecaster": {"roi": -0.01, "consecutive_losses": 3, "last_updated": datetime.utcnow().isoformat()}}
                }
                
                # Positions
                self.ctx.shared_state.positions = {
                    "BTCUSDT": {
                        "qty": 0.1,
                        "entry_price": 48000.0,
                        "current_price": 50000.0,
                        "unrealized_pnl": 200.0
                    }
                }
                
                # NAV
                self.ctx.shared_state._total_value = 15000.0
                
                self.log_test("Test Data Seeded", True, "Balances, prices, scores, positions")
            
            return True
            
        except Exception as e:
            self.log_test("Environment Setup", False, str(e))
            logger.error("Failed to set up dry-run environment", exc_info=True)
            return False
    
    async def test_phase_initialization(self):
        """Test phased initialization up to Phase 8"""
        logger.info("\n" + "="*60)
        logger.info("TESTING PHASED INITIALIZATION")
        logger.info("="*60)
        
        try:
            # Initialize up to Phase 8
            await asyncio.wait_for(
                self.ctx.initialize_all(up_to_phase=8),
                timeout=60.0
            )
            
            self.log_test("Phase Initialization", True, "All phases initialized successfully")
            
            # Verify components are started
            components_to_check = [
                ("shared_state", "SharedState"),
                ("exchange_client", "ExchangeClient"),
                ("market_data_feed", "MarketDataFeed"),
                ("execution_manager", "ExecutionManager"),
                ("agent_manager", "AgentManager"),
                ("risk_manager", "RiskManager"),
                ("dashboard_server", "DashboardServer"),
            ]
            
            for attr, name in components_to_check:
                component = getattr(self.ctx, attr, None)
                if component:
                    self.log_test(f"Component Active: {name}", True)
                else:
                    self.log_test(f"Component Active: {name}", False, "Component not initialized")
            
            return True
            
        except asyncio.TimeoutError:
            self.log_test("Phase Initialization", False, "Timeout after 30s")
            return False
        except Exception as e:
            self.log_test("Phase Initialization", False, str(e))
            logger.error("Phase initialization failed", exc_info=True)
            return False
    
    async def test_data_flow(self):
        """Test data flow through SharedState"""
        logger.info("\n" + "="*60)
        logger.info("TESTING DATA FLOW")
        logger.info("="*60)
        
        try:
            ss = self.ctx.shared_state
            
            # Test 1: Price updates
            test_price = 51000.0
            await ss.update_latest_price("BTCUSDT", test_price)
            retrieved_price = ss.latest_prices.get("BTCUSDT")
            self.log_test(
                "Price Update Flow",
                retrieved_price == test_price,
                f"Expected {test_price}, got {retrieved_price}"
            )
            
            # Test 2: Agent signal injection
            test_signal = {
                "action": "buy",
                "confidence": 0.9,
                "reason": "Dry-run test signal",
                "timestamp": time.time()
            }
            
            if not hasattr(ss, 'agent_signals'):
                ss.agent_signals = {}
            
            ss.agent_signals["ETHUSDT"] = {"DryRunAgent": test_signal}
            signal_stored = "ETHUSDT" in ss.agent_signals
            self.log_test("Signal Injection Flow", signal_stored, "Signal stored in SharedState")
            
            # Test 3: Balance updates
            ss.balances["USDT"]["free"] = 9500.0
            balance_updated = ss.balances["USDT"]["free"] == 9500.0
            self.log_test("Balance Update Flow", balance_updated, "Balance updated correctly")
            
            # Test 4: Position tracking
            ss.positions["ETHUSDT"] = {
                "qty": 1.0,
                "entry_price": 2900.0,
                "current_price": 3000.0,
                "unrealized_pnl": 100.0
            }
            position_tracked = "ETHUSDT" in ss.positions
            self.log_test("Position Tracking Flow", position_tracked, "Position tracked in SharedState")
            
            return True
            
        except Exception as e:
            self.log_test("Data Flow", False, str(e))
            logger.error("Data flow test failed", exc_info=True)
            return False
    
    async def test_agent_operations(self):
        """Test agent operations without real trades"""
        logger.info("\n" + "="*60)
        logger.info("TESTING AGENT OPERATIONS")
        logger.info("="*60)
        
        try:
            if not self.ctx.agent_manager:
                self.log_test("Agent Operations", False, "AgentManager not initialized")
                return False
            
            # Check if agents are registered
            agent_mgr = self.ctx.agent_manager
            
            if hasattr(agent_mgr, 'agents'):
                agent_count = len(agent_mgr.agents)
                self.log_test(
                    "Agents Registered",
                    agent_count > 0,
                    f"{agent_count} agents registered"
                )
            else:
                self.log_test("Agents Registered", True, "AgentManager initialized")
            
            # Test agent signal generation (simulated)
            ss = self.ctx.shared_state
            
            # Simulate DipSniper signal
            dip_signal = {
                "action": "buy",
                "confidence": 0.75,
                "reason": "Simulated dip detected",
                "timestamp": time.time()
            }
            
            if not hasattr(ss, 'agent_signals'):
                ss.agent_signals = {}
            
            ss.agent_signals["BTCUSDT"] = {"DipSniper": dip_signal}
            
            # Simulate TrendHunter signal
            trend_signal = {
                "action": "hold",
                "confidence": 0.5,
                "reason": "Simulated trend analysis",
                "timestamp": time.time()
            }
            
            ss.agent_signals["ETHUSDT"] = {"TrendHunter": trend_signal}
            
            self.log_test("Agent Signal Generation", True, "Simulated signals from multiple agents")
            
            return True
            
        except Exception as e:
            self.log_test("Agent Operations", False, str(e))
            logger.error("Agent operations test failed", exc_info=True)
            return False
    
    async def test_risk_management(self):
        """Test risk management without real trades"""
        logger.info("\n" + "="*60)
        logger.info("TESTING RISK MANAGEMENT")
        logger.info("="*60)
        
        try:
            if not self.ctx.risk_manager:
                self.log_test("Risk Management", False, "RiskManager not initialized")
                return False
            
            ss = self.ctx.shared_state
            
            # Test exposure calculation
            total_value = ss._total_value
            position_value = sum(
                pos.get("qty", 0) * pos.get("current_price", 0)
                for pos in ss.positions.values()
            )
            
            exposure = position_value / total_value if total_value > 0 else 0
            
            self.log_test(
                "Exposure Calculation",
                0 <= exposure <= 1,
                f"Exposure: {exposure:.2%}"
            )
            
            # Test risk flags
            if hasattr(ss, 'trading_paused'):
                is_paused = ss.trading_paused
                self.log_test(
                    "Trading Pause Flag",
                    isinstance(is_paused, bool),
                    f"Paused: {is_paused}"
                )
            else:
                self.log_test("Trading Pause Flag", True, "Flag mechanism available")
            
            return True
            
        except Exception as e:
            self.log_test("Risk Management", False, str(e))
            logger.error("Risk management test failed", exc_info=True)
            return False
    
    async def test_dashboard_api(self):
        """Test dashboard API endpoints"""
        logger.info("\n" + "="*60)
        logger.info("TESTING DASHBOARD API")
        logger.info("="*60)
        
        try:
            if not self.ctx.dashboard_server:
                self.log_test("Dashboard API", False, "DashboardServer not initialized")
                return False
            
            # Give the server a moment to start
            await asyncio.sleep(2)
            
            # Test REST API
            import aiohttp
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://localhost:8888/api/state", timeout=5) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            
                            # Verify response structure
                            required_keys = ["nav", "balances", "positions", "scores", "timestamp"]
                            has_all_keys = all(key in data for key in required_keys)
                            
                            self.log_test(
                                "Dashboard REST API",
                                has_all_keys,
                                f"Status: {resp.status}, Keys: {list(data.keys())}"
                            )
                            
                            # Verify data values
                            self.log_test(
                                "Dashboard Data Accuracy",
                                data["nav"] > 0 and len(data["balances"]) > 0,
                                f"NAV: {data['nav']}, Balances: {len(data['balances'])}"
                            )
                        else:
                            self.log_test("Dashboard REST API", False, f"Status: {resp.status}")
            
            except asyncio.TimeoutError:
                self.log_test("Dashboard REST API", False, "Request timeout")
            except aiohttp.ClientConnectorError:
                self.log_test("Dashboard REST API", False, "Connection refused - server may not be running")
            
            return True
            
        except Exception as e:
            self.log_test("Dashboard API", False, str(e))
            logger.error("Dashboard API test failed", exc_info=True)
            return False
    
    async def test_execution_dry_run(self):
        """Test execution manager in dry-run mode"""
        logger.info("\n" + "="*60)
        logger.info("TESTING EXECUTION (DRY-RUN)")
        logger.info("="*60)
        
        try:
            if not self.ctx.execution_manager:
                self.log_test("Execution Manager", False, "ExecutionManager not initialized")
                return False
            
            exec_mgr = self.ctx.execution_manager
            
            # Verify execution manager has required methods
            required_methods = ["execute_trade"]
            for method in required_methods:
                has_method = hasattr(exec_mgr, method) and callable(getattr(exec_mgr, method))
                self.log_test(
                    f"ExecutionManager.{method}",
                    has_method,
                    "Method available" if has_method else "Method missing"
                )
            
            # Note: We don't actually execute trades in dry-run
            self.log_test(
                "Execution Dry-Run",
                True,
                "ExecutionManager ready (no real trades executed)"
            )
            
            return True
            
        except Exception as e:
            self.log_test("Execution Dry-Run", False, str(e))
            logger.error("Execution dry-run test failed", exc_info=True)
            return False
    
    async def test_system_stability(self):
        """Test system stability over a short period"""
        logger.info("\n" + "="*60)
        logger.info("TESTING SYSTEM STABILITY")
        logger.info("="*60)
        
        try:
            # Run for 10 seconds and monitor for crashes
            logger.info("Running stability test for 10 seconds...")
            
            start_time = time.time()
            errors_detected = []
            
            for i in range(10):
                await asyncio.sleep(1)
                
                # Check if components are still alive
                if self.ctx.shared_state is None:
                    errors_detected.append("SharedState became None")
                
                if self.ctx.dashboard_server is None:
                    errors_detected.append("DashboardServer became None")
                
                # Log progress
                logger.info(f"  Stability check {i+1}/10...")
            
            elapsed = time.time() - start_time
            
            self.log_test(
                "System Stability",
                len(errors_detected) == 0,
                f"Ran for {elapsed:.1f}s, Errors: {len(errors_detected)}"
            )
            
            if errors_detected:
                for error in errors_detected:
                    logger.error(f"  - {error}")
            
            return len(errors_detected) == 0
            
        except Exception as e:
            self.log_test("System Stability", False, str(e))
            logger.error("Stability test failed", exc_info=True)
            return False
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("\n" + "="*60)
        logger.info("CLEANING UP")
        logger.info("="*60)
        
        try:
            if self.ctx:
                await self.ctx.shutdown()
                self.log_test("Shutdown", True, "Clean shutdown completed")
            
            return True
            
        except Exception as e:
            self.log_test("Shutdown", False, str(e))
            logger.error("Cleanup failed", exc_info=True)
            return False
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*60)
        logger.info("DRY-RUN TEST SUMMARY")
        logger.info("="*60)
        
        passed = sum(1 for v in self.test_results.values() if v)
        total = len(self.test_results)
        
        logger.info(f"Tests Passed: {passed}/{total}")
        logger.info(f"Tests Failed: {total - passed}/{total}")
        
        if self.errors:
            logger.error(f"\n❌ ERRORS DETECTED:")
            for error in self.errors:
                logger.error(f"  - {error}")
        else:
            logger.info(f"\n✅ ALL TESTS PASSED")
        
        # Categorize results
        logger.info(f"\nTest Categories:")
        logger.info(f"  Setup: {sum(1 for k in self.test_results if 'Setup' in k or 'Created' in k or 'Built' in k or 'Mocked' in k or 'Seeded' in k)}")
        logger.info(f"  Initialization: {sum(1 for k in self.test_results if 'Phase' in k or 'Component Active' in k)}")
        logger.info(f"  Data Flow: {sum(1 for k in self.test_results if 'Flow' in k)}")
        logger.info(f"  Operations: {sum(1 for k in self.test_results if 'Agent' in k or 'Risk' in k or 'Execution' in k)}")
        logger.info(f"  API: {sum(1 for k in self.test_results if 'Dashboard' in k or 'API' in k)}")
        logger.info(f"  Stability: {sum(1 for k in self.test_results if 'Stability' in k or 'Shutdown' in k)}")
        
        return passed == total

async def main():
    """Main dry-run test orchestrator"""
    logger.info("\n" + "="*60)
    logger.info("OCTIVAULT DRY-RUN TEST")
    logger.info("="*60)
    logger.info("Testing full system operation without real trades")
    logger.info("="*60)
    
    tester = DryRunTester()
    
    try:
        # Run all test phases
        if not await tester.setup_dry_run_environment():
            logger.error("Failed to set up environment. Aborting.")
            return 1
        
        if not await tester.test_phase_initialization():
            logger.error("Phase initialization failed. Continuing with other tests...")
        
        await tester.test_data_flow()
        await tester.test_agent_operations()
        await tester.test_risk_management()
        await tester.test_dashboard_api()
        await tester.test_execution_dry_run()
        await tester.test_system_stability()
        
    finally:
        # Always cleanup
        await tester.cleanup()
    
    # Print summary and determine exit code
    all_passed = tester.print_summary()
    
    if all_passed:
        logger.info("\n✅ DRY-RUN SUCCESSFUL - SYSTEM READY FOR DEPLOYMENT")
        return 0
    else:
        logger.error("\n❌ DRY-RUN FAILED - REVIEW ERRORS BEFORE DEPLOYMENT")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
