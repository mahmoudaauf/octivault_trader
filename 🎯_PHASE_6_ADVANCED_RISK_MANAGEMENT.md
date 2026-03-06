# 🎯 Phase 6: Advanced Risk Management & System Hardening

**Date**: March 6, 2026  
**Status**: ✅ **INITIALIZED**  
**Objective**: Transition from Phase 5's pre-trade concentration gating to comprehensive advanced risk management  

---

## Phase Transition: Phase 5 → Phase 6

### Phase 5 Achievements (Now Complete)
✅ **Pre-Trade Concentration Risk Gating**
- Concentration limits enforced BEFORE trades execute
- Eliminates oversized positions entering market
- Zero deadlock-class bugs

✅ **Core Systems Integrated**
- Position sizing with headroom awareness
- Bootstrap metrics persistence
- Dust registry lifecycle management
- Position merger & consolidation
- Trading coordinator unified interface

✅ **Test Coverage**
- 37 integration tests passing
- All Phase 1-5 components verified
- Trading coordinator fully operational

### Phase 6 Mission
Build on Phase 5's solid foundation to add:
1. **Advanced Position Risk Analysis**
2. **Portfolio-level Risk Aggregation**
3. **Dynamic Risk Limit Adjustment**
4. **Margin & Leverage Management**
5. **Scenario Analysis & Stress Testing**
6. **Risk Dashboard & Real-time Monitoring**

---

## Phase 6 Architecture Overview

### Component Structure

```
Phase 6: Advanced Risk Management
│
├─ Risk Analysis Engine (NEW)
│  ├─ Position risk metrics (PnL, Greeks, Value at Risk)
│  ├─ Portfolio aggregation (total risk, correlation analysis)
│  └─ Risk attribution (by symbol, by bracket, by strategy)
│
├─ Dynamic Risk Limits (NEW)
│  ├─ Per-symbol exposure limits
│  ├─ Portfolio-wide concentration caps
│  ├─ Bracket-adaptive limits (MICRO → LARGE)
│  └─ Real-time adjustment based on volatility
│
├─ Margin & Leverage Manager (NEW)
│  ├─ Margin requirement calculation
│  ├─ Leverage utilization tracking
│  ├─ Liquidation risk monitoring
│  └─ Forced position reduction triggers
│
├─ Scenario & Stress Testing (NEW)
│  ├─ Historical scenario playback
│  ├─ Flash crash simulation
│  ├─ Portfolio liquidation testing
│  └─ Recovery path analysis
│
├─ Risk Dashboard (NEW)
│  ├─ Real-time risk metrics display
│  ├─ Alert & notification system
│  ├─ Risk trending & analytics
│  └─ Compliance reporting
│
└─ Integration with Phase 1-5
   ├─ Capital Governor (position sizing)
   ├─ Dust Registry (position tracking)
   ├─ Position Merger (consolidation)
   └─ Trading Coordinator (execution gating)
```

---

## Implementation Roadmap

### Phase 6.1: Position Risk Metrics (Weeks 1-2)
**Goal**: Calculate comprehensive risk metrics for each position

#### 1.1 Position Risk Calculator
```python
class PositionRiskCalculator:
    """Calculate risk metrics for individual positions."""
    
    def calculate_position_risk(
        self,
        position: Dict[str, Any],
        market_data: Dict[str, float],
        volatility: float
    ) -> Dict[str, float]:
        """
        Calculate position risk metrics.
        
        Returns:
            {
                "notional_value": float,        # Position value in USD
                "pnl_unrealized": float,        # Unrealized P&L
                "pnl_percent": float,           # P&L as % of position
                "vat_95": float,                # Value at Risk (95% confidence)
                "duration": float,              # Sensitivity to 1% move
                "marginal_var": float,          # Contribution to portfolio VaR
                "max_loss": float,              # Maximum loss scenario (static)
            }
        """
```

#### 1.2 Value at Risk (VaR) Calculation
```python
def calculate_var(
    position_value: float,
    volatility: float,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1
) -> float:
    """
    Calculate Value at Risk using parametric method.
    
    VaR = Position Value × Z-score × Volatility × √Time
    
    - Position Value: Dollar exposure
    - Z-score: 1.645 for 95% confidence, 2.326 for 99%
    - Volatility: Annualized price volatility
    - Time: Fraction of year (1 day = 1/252)
    """
```

#### 1.3 Greeks Calculation (For Options-like Exposure)
```python
def estimate_greeks(
    position: Dict,
    market_data: Dict,
    volatility: float
) -> Dict[str, float]:
    """
    Estimate option-like Greeks for position sensitivity.
    
    Returns:
        {
            "delta": float,      # Change in position value per $1 price move
            "gamma": float,      # Change in delta per $1 price move
            "vega": float,       # Change per 1% volatility move
            "theta": float,      # Daily decay (if time-dependent)
        }
    """
```

---

### Phase 6.2: Portfolio Risk Aggregation (Weeks 2-3)
**Goal**: Combine position risks into portfolio-level metrics

#### 2.1 Portfolio Risk Aggregator
```python
class PortfolioRiskAggregator:
    """Aggregate position risks at portfolio level."""
    
    def aggregate_risks(
        self,
        positions: List[Dict],
        market_data: Dict,
        volatility_by_symbol: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Aggregate all position risks.
        
        Returns:
            {
                "total_notional": float,        # Sum of all positions
                "total_pnl": float,             # Sum of unrealized P&L
                "portfolio_var_95": float,      # Portfolio VaR (95%)
                "concentration_index": float,   # Herfindahl index (0-1)
                "max_single_loss": float,       # Largest potential loss
                "correlation_adjusted_var": float,  # Adjusted for correlations
                "beta_weighted_var": float,     # Market sensitivity-weighted VaR
                "by_symbol": {...},             # Per-symbol breakdown
                "by_bracket": {...},            # MICRO/SMALL/MEDIUM/LARGE breakdown
            }
        """
```

#### 2.2 Correlation Analysis
```python
def estimate_correlations(
    price_history: Dict[str, List[float]],
    lookback_periods: int = 60
) -> Dict[str, Dict[str, float]]:
    """
    Estimate rolling correlations between symbols.
    
    Used for portfolio VaR adjustment (diversification benefit).
    Positive correlation → less diversification → higher VaR
    Negative correlation → better diversification → lower VaR
    """
```

#### 2.3 Concentration Index (Herfindahl)
```python
def calculate_concentration_index(
    position_values: Dict[str, float]
) -> float:
    """
    Calculate Herfindahl concentration index.
    
    Formula: HHI = Σ (position_i / total)²
    
    - 0.00 = Perfect diversification (impossible)
    - 0.01 = Well diversified
    - 0.10 = Moderately concentrated
    - 1.00 = All in one position
    
    Used to trigger dynamic risk limit adjustments.
    """
```

---

### Phase 6.3: Dynamic Risk Limits (Weeks 3-4)
**Goal**: Adjust position sizing & concentration limits based on system state

#### 3.1 Dynamic Risk Limit Manager
```python
class DynamicRiskLimitManager:
    """Adjust risk limits based on portfolio state."""
    
    def calculate_risk_limits(
        self,
        nav: float,
        portfolio_risk: Dict,
        market_volatility: Dict,
        system_stress_level: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate adjusted position size limits.
        
        Factors considered:
        1. Current NAV (capital available)
        2. Portfolio concentration
        3. Market volatility (increase limits in calm markets)
        4. System stress level (decrease limits during stress)
        5. Historical drawdown
        
        Returns:
            {
                "max_position_by_symbol": {...},        # Per-symbol limits
                "max_concentration_pct": float,         # Portfolio max concentration
                "max_leverage_allowed": float,          # Margin utilization limit
                "emergency_halt_threshold": float,      # Circuit breaker level
                "stress_level_adjustment": float,       # 0.5x to 1.5x multiplier
            }
        """
```

#### 3.2 Bracket-Adaptive Limits
```python
def get_bracket_risk_limits(
    nav: float,
    volatility: float,
    stress_level: float
) -> Dict[str, Dict[str, float]]:
    """
    Return risk limits by NAV bracket with volatility adjustment.
    
    MICRO (< $500):
        Base: 50% concentration, 5x leverage
        High vol: 35% concentration, 3x leverage
        Stress: 20% concentration, 1.5x leverage
    
    SMALL ($500-2K):
        Base: 35% concentration, 4x leverage
        High vol: 25% concentration, 2.5x leverage
        Stress: 15% concentration, 1.2x leverage
    
    MEDIUM ($2K-10K):
        Base: 25% concentration, 3x leverage
        High vol: 18% concentration, 2x leverage
        Stress: 10% concentration, 1x leverage
    
    LARGE ($10K+):
        Base: 20% concentration, 2x leverage
        High vol: 15% concentration, 1.5x leverage
        Stress: 8% concentration, 1x leverage
    """
```

#### 3.3 Stress Level Calculation
```python
def calculate_system_stress_level() -> float:
    """
    Calculate system stress on 0.0 (calm) to 1.0 (emergency) scale.
    
    Factors:
    - Portfolio drawdown from high water mark
    - Volatility spike vs. 30-day average
    - Number of positions in dust state
    - Failed trade attempts / rejections
    - Position liquidation events
    
    Threshold-based actions:
    - 0.0-0.3: Normal operation (100% limits)
    - 0.3-0.6: Elevated caution (50-75% limits)
    - 0.6-0.8: High stress (25-50% limits)
    - 0.8-1.0: Emergency (minimal trading, risk reduction focus)
    """
```

---

### Phase 6.4: Margin & Leverage Management (Weeks 4-5)
**Goal**: Track margin utilization and prevent over-leverage

#### 4.1 Margin Calculator
```python
class MarginManager:
    """Calculate and manage margin requirements."""
    
    def calculate_margin_requirement(
        self,
        positions: List[Dict],
        exchange_rules: Dict,
        margin_ratio: float = 0.1
    ) -> Dict[str, float]:
        """
        Calculate total margin required for positions.
        
        Margin = Σ (position_notional × margin_ratio)
        
        Returns:
            {
                "total_margin_required": float,   # Total margin needed
                "margin_available": float,        # Account balance - reserved
                "margin_utilization": float,      # Required / Available
                "excess_margin": float,           # Available - Required
                "margin_warning_level": float,    # 70% of available
                "margin_liquidation_level": float,  # 80% of available
            }
        """
```

#### 4.2 Leverage Limiter
```python
class LeverageLimiter:
    """Limit leverage to safe levels."""
    
    def get_max_leverage(
        self,
        account_balance: float,
        portfolio_risk: Dict,
        stress_level: float
    ) -> float:
        """
        Calculate maximum safe leverage multiplier.
        
        Base leverage: 3x (account balance)
        Risk adjustment: VaR-based reduction
        Stress adjustment: Reduced during high stress
        
        Examples:
        - Account: $1000, leverage 3x → max position $3000
        - Account: $1000, leverage 1x → max position $1000
        """
```

#### 4.3 Forced Liquidation Risk
```python
def calculate_liquidation_risk(
    positions: List[Dict],
    margin_level: float,
    margin_ratio: float = 0.1
) -> Dict[str, Any]:
    """
    Calculate risk of forced liquidation.
    
    Returns:
        {
            "liquidation_probability": float,   # Based on historical drawdown
            "price_move_to_liquidation": float, # How far prices need to move
            "positions_at_risk": List[str],     # Symbols most likely to be liquidated
            "recovery_plan": {...},             # Suggested actions to reduce risk
        }
    """
```

---

### Phase 6.5: Scenario & Stress Testing (Weeks 5-6)
**Goal**: Test portfolio resilience under adverse scenarios

#### 5.1 Scenario Analyzer
```python
class ScenarioAnalyzer:
    """Run historical and hypothetical scenarios."""
    
    def run_scenario(
        self,
        positions: List[Dict],
        scenario_name: str,
        price_moves: Dict[str, float],
        volatility_multiplier: float = 1.0
    ) -> Dict[str, Any]:
        """
        Simulate portfolio under scenario.
        
        Scenarios:
        - "crash_2020": COVID collapse (-30%)
        - "flash_crash": Intraday spike (±15%)
        - "black_monday": Oct 1987 (-22% single day)
        - "rate_shock": Fed surprise (varies by asset class)
        - "correlation_breakdown": All positions move same direction
        
        Returns:
            {
                "scenario_pnl": float,          # Portfolio P&L in scenario
                "max_drawdown": float,          # Worst-case loss
                "recovery_time": int,           # Days to recover to breakeven
                "positions_liquidated": List,   # Which positions force-sold
                "margin_call_triggered": bool,  # Margin sufficient?
            }
        """
```

#### 5.2 Historical Backtester
```python
def backtest_historical_scenario(
    positions: List[Dict],
    date_range: Tuple[datetime, datetime],
    historical_data: Dict
) -> Dict[str, Any]:
    """
    Replay portfolio through historical period.
    
    Examples:
    - "COVID collapse" (Feb-Mar 2020)
    - "Crypto winter" (Nov-Dec 2022)
    - "Taper tantrum" (May-Jun 2013)
    
    Returns:
        Daily equity curve, drawdown analysis, recovery metrics.
    """
```

#### 5.3 Stress Test Report
```python
def generate_stress_test_report(
    positions: List[Dict],
    scenarios: List[str],
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Generate comprehensive stress test report.
    
    Outputs:
    - Worst-case P&L by scenario
    - Required capital to survive all scenarios
    - Risk of portfolio ruin
    - Recommended position reductions
    """
```

---

### Phase 6.6: Risk Dashboard & Monitoring (Weeks 6-7)
**Goal**: Real-time risk monitoring and alerting

#### 6.1 Risk Metrics Tracker
```python
class RiskMetricsTracker:
    """Track and trend risk metrics over time."""
    
    async def publish_risk_metrics(self) -> Dict[str, Any]:
        """
        Publish current risk snapshot to dashboard.
        
        Returns:
            {
                "timestamp": float,
                "portfolio": {
                    "nav": float,
                    "pnl": float,
                    "var_95": float,
                    "concentration": float,
                    "leverage_ratio": float,
                    "stress_level": float,
                },
                "positions": [
                    {
                        "symbol": str,
                        "notional": float,
                        "pnl": float,
                        "var": float,
                        "risk_contribution": float,
                    }
                ],
                "alerts": [
                    {
                        "severity": "WARNING|CRITICAL",
                        "message": str,
                        "affected_symbol": str,
                    }
                ]
            }
        """
```

#### 6.2 Alert Engine
```python
class RiskAlertEngine:
    """Generate alerts based on risk thresholds."""
    
    async def check_risk_thresholds(self) -> List[Alert]:
        """
        Check against thresholds and generate alerts.
        
        Thresholds:
        - Position size > max limit
        - Portfolio concentration > 50%
        - VaR > 10% of NAV (WARNING) / 15% (CRITICAL)
        - Leverage > 2x (WARNING) / 3x (CRITICAL)
        - Margin utilization > 70% (WARNING) / 80% (CRITICAL)
        - Liquidation risk > 25% (CRITICAL)
        """
```

#### 6.3 Risk Analytics
```python
class RiskAnalytics:
    """Generate risk analytics for reporting."""
    
    def generate_daily_risk_report(self) -> Dict[str, Any]:
        """
        Generate end-of-day risk report.
        
        Includes:
        - Risk metrics snapshot
        - Daily risk activity (trades, adjustments)
        - Exposure breakdown
        - Stress test results
        - Compliance metrics
        """
    
    def generate_risk_attribution(self) -> Dict[str, float]:
        """
        Attribute portfolio risk to sources.
        
        Breakdown:
        - By symbol (BTC, ETH, etc.)
        - By bracket (MICRO, SMALL, MEDIUM, LARGE)
        - By risk factor (volatility, concentration, leverage)
        """
```

---

### Phase 6.7: Integration with Trading Coordinator (Week 7)
**Goal**: Connect risk management to trade execution

#### 7.1 Risk-Aware Trade Gating
```python
def pre_trade_risk_check(
    symbol: str,
    quantity: float,
    price: float,
    trade_type: str,
    risk_manager: RiskManager
) -> Tuple[bool, str]:
    """
    Check if trade respects risk limits before execution.
    
    Checks:
    1. Position size within limit
    2. Portfolio concentration within limit
    3. Leverage within safe bounds
    4. Margin available
    5. No emergency/stress halt active
    
    Returns:
        (approved: bool, reason: str)
    """
```

#### 7.2 Risk Limit Enforcement in Trading Coordinator
```python
# In TradingCoordinator.execute_trade()
def execute_trade_with_risk_gates(
    self,
    symbol: str,
    positions: List[Dict],
    order_params: Dict
) -> Optional[str]:
    """
    Execute trade with risk gates (Phase 5) + advanced risk checks (Phase 6).
    
    1. Check system readiness (Phase 5)
    2. Check concentration risk (Phase 5)
    3. Check position risk limits (Phase 6 NEW)
    4. Check portfolio risk (Phase 6 NEW)
    5. Check margin requirements (Phase 6 NEW)
    6. Execute trade if all pass
    7. Update risk metrics
    """
```

---

## Risk Limit Framework

### Risk Metric Definitions

| Metric | Definition | Unit | Use Case |
|--------|-----------|------|----------|
| **Notional Value** | Position value in USD | $ | Exposure sizing |
| **VaR 95%** | 95% confidence max loss | $ | Risk quantification |
| **VaR 99%** | 99% confidence max loss | $ | Stress testing |
| **Expected Shortfall** | Average loss if VaR exceeded | $ | Tail risk |
| **Concentration Index** | Herfindahl measure (0-1) | ratio | Diversification |
| **Margin Ratio** | Required margin / Account | % | Leverage safety |
| **Liquidation Distance** | Price move to liquidation | % | Forced sale risk |
| **Stress Test Return** | Portfolio return in scenario | % | Scenario resilience |

### Risk Decision Tree

```
Is trade accepted?
│
├─ System Ready? (Phase 5)
│  └─ NO → REJECT ("System not ready")
│  └─ YES → Continue
│
├─ Concentration Gate Passed? (Phase 5)
│  └─ NO → REJECT ("Concentration limit exceeded")
│  └─ YES → Continue (Phase 6)
│
├─ Position Risk within Limit? (Phase 6 NEW)
│  └─ NO → REJECT ("Position size exceeds limit")
│  └─ YES → Continue
│
├─ Portfolio Risk acceptable? (Phase 6 NEW)
│  └─ NO → REJECT ("Portfolio risk too high")
│  └─ YES → Continue
│
├─ Margin Available? (Phase 6 NEW)
│  └─ NO → REJECT ("Insufficient margin")
│  └─ YES → Continue
│
├─ Emergency Halt Active? (Phase 6 NEW)
│  └─ YES → REJECT ("Emergency trading halt")
│  └─ NO → Continue
│
└─ ACCEPT ✅ → Execute Trade
   └─ Update Risk Metrics
   └─ Check Alerts
   └─ Publish Dashboard
```

---

## Success Criteria

### Functional Requirements
✓ All position risks calculated in real-time  
✓ Portfolio risk aggregated within 100ms  
✓ Dynamic limits adjust to market conditions  
✓ Margin requirements enforced pre-trade  
✓ Stress tests run daily  
✓ Dashboard updates every 1 second  
✓ Alerts generated and published within 1 second  
✓ Zero margin calls (forced liquidations prevented)  

### Performance Requirements
✓ Risk calculation < 50ms per position  
✓ Portfolio aggregation < 100ms  
✓ Scenario analysis < 500ms  
✓ Dashboard update < 1 second  
✓ Alert generation < 100ms  

### Risk Requirements
✓ Portfolio VaR never exceeds 15% of NAV  
✓ Concentration never exceeds 50% (emergency) / 30% (normal)  
✓ Leverage never exceeds 3x (emergency) / 2x (normal)  
✓ Margin utilization never exceeds 80%  
✓ Liquidation risk always < 10%  

---

## Deployment Checklist

- [ ] Position Risk Calculator implemented
- [ ] Portfolio Aggregator tested with 100+ positions
- [ ] Dynamic Limit Manager optimized
- [ ] Margin Manager integrated
- [ ] Scenario Analysis validated against historical data
- [ ] Risk Dashboard built & tested
- [ ] Alert Engine deployed
- [ ] Trading Coordinator integrated with risk gates
- [ ] End-to-end integration tests passing
- [ ] Performance targets met
- [ ] Production deployment ready

---

## What's Next (Phase 7)

After Phase 6 completes, Phase 7 will focus on:
1. **Machine Learning Risk Models** - Predictive VaR, anomaly detection
2. **Network Risk Analysis** - Counterparty & systemic risk
3. **Compliance Automation** - Regulatory reporting, audit trails
4. **Advanced Portfolio Optimization** - Mean-variance, risk parity rebalancing

---

**End of Phase 6 Documentation**
