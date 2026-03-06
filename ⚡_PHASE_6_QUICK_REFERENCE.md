# ⚡ Phase 6 Quick Reference & Fast-Track Implementation

**Status**: Ready to Build  
**Timeline**: 7 weeks (2-3 weeks per major component)  
**Starting Point**: Phase 5 ✅ Complete (37 tests passing)

---

## Core Phase 6 Components (TL;DR)

### 1️⃣ Position Risk Calculator
```python
# Calculate risk for one position
position_risk = calculator.calculate_position_risk(
    position={"symbol": "BTC", "quantity": 0.5},
    market_data={"BTC": {"price": 50000, "volatility": 0.25}},
    volatility=0.25
)
# Returns: notional_value, pnl, vat_95, max_loss, etc.
```

### 2️⃣ Portfolio Aggregator
```python
# Get total portfolio risk
portfolio_risk = aggregator.aggregate_risks(
    positions=all_positions,
    market_data=current_prices,
    volatility_by_symbol=vol_matrix
)
# Returns: total_var, concentration, by_symbol breakdown
```

### 3️⃣ Dynamic Risk Limits
```python
# Get adjusted position size limits
limits = limit_manager.calculate_risk_limits(
    nav=account_nav,
    portfolio_risk=current_risk,
    market_volatility=vol_data,
    system_stress_level=0.0  # 0=calm, 1=emergency
)
# Returns: max_position_by_symbol, concentration limits, leverage limits
```

### 4️⃣ Margin Manager
```python
# Check if trade is margin-safe
margin_status = margin_manager.check_margin(
    positions=all_positions,
    proposed_trade=new_position,
    account_balance=nav
)
# Returns: margin_required, margin_available, safe=True/False
```

### 5️⃣ Stress Tester
```python
# Test portfolio under crash scenario
result = stress_tester.run_scenario(
    positions=all_positions,
    scenario_name="crash_2020",  # COVID collapse
    price_moves={"BTC": -0.30, "ETH": -0.35}
)
# Returns: portfolio_pnl, max_drawdown, recovery_time
```

### 6️⃣ Risk Dashboard
```python
# Publish risk metrics in real-time
metrics = {
    "nav": 50000,
    "portfolio_var": 7500,          # $7,500 at risk
    "concentration": 0.35,          # 35% in one position
    "leverage": 1.5,                # 1.5x leveraged
    "stress_level": 0.2,            # 20% stress (calm)
}
# Display on dashboard, generate alerts if thresholds exceeded
```

---

## Key Formulas (Cheat Sheet)

### Value at Risk (VaR)
```
VaR = Position Value × Z-score × Volatility × √Time

Example: $10,000 position, 25% annual volatility, 1 day
VaR = $10,000 × 1.645 × 0.25 × √(1/252)
VaR = $10,000 × 1.645 × 0.25 × 0.063
VaR ≈ $259 (95% confidence)
```

### Herfindahl Concentration Index
```
HHI = Σ (position_i / total_portfolio)²

Example: $100 portfolio with $50 in BTC, $30 in ETH, $20 in ADA
HHI = (0.50)² + (0.30)² + (0.20)²
HHI = 0.25 + 0.09 + 0.04 = 0.38 (moderately concentrated)
```

### Margin Requirement
```
Margin Required = Σ(Position Notional × Margin Ratio)

Example: 2.0 BTC @ $50,000, 1% margin ratio
Margin = (2 × $50,000) × 0.01 = $1,000
```

### Liquidation Distance
```
Liquidation Price = (Margin × Account Balance) / Position Notional

Example: $1000 margin required, $50,000 account, $100,000 position
Liquidation when price moves: ($1000 / $100,000) = 1% down
```

---

## Integration Points with Phases 1-5

### Phase 5 → Phase 6 Handoff

**Phase 5 (Pre-Trade Concentration Gate)**
```
Trade Size Limit = Headroom
Headroom = (NAV × Max%) - Current Position
Quote Capped at Headroom
```

**Phase 6 Enhancement (Pre-Trade Risk Gate)**
```
Additional Checks:
1. Position VaR < Limit
2. Portfolio VaR < Limit  
3. Margin Available
4. Leverage < Max
5. Not in Emergency Halt
```

---

## Testing Strategy

### Unit Tests
```
✓ Position risk calculator (10 positions)
✓ Portfolio aggregator (50-position portfolio)
✓ Dynamic limits calculation
✓ Margin manager logic
✓ Stress test scenarios
```

### Integration Tests
```
✓ Risk-aware trade execution
✓ Alert generation pipeline
✓ Dashboard data flow
✓ End-to-end risk gate enforcement
```

### Load Tests
```
✓ 1000+ positions risk aggregation < 100ms
✓ Dashboard update every 1 second
✓ Alert generation < 100ms
```

---

## Quick Start: First Implementation (Week 1)

### Files to Create
```
core/risk_calculator.py           # Position risk metrics
core/risk_aggregator.py           # Portfolio aggregation
core/risk_limit_manager.py        # Dynamic limits
core/margin_manager.py            # Margin enforcement
core/stress_tester.py             # Scenario analysis
```

### Minimum Viable Phase 6
```
1. Position VaR calculation
2. Portfolio VaR aggregation
3. Dynamic position size limits
4. Pre-trade risk check
5. Basic alert system
```

### Test Coverage Target
```
✓ 100+ new integration tests
✓ All risk gates passing
✓ Performance < 100ms for portfolio calc
✓ Stress tests validated against historical data
```

---

## Risk Thresholds (Starter Config)

| Metric | Normal | Warning | Critical | Action |
|--------|--------|---------|----------|--------|
| **Portfolio VaR** | <10% NAV | 10-15% | >15% | Reduce size |
| **Concentration** | <30% | 30-50% | >50% | Diversify |
| **Leverage** | <2x | 2-2.5x | >2.5x | Reduce leverage |
| **Margin Util** | <70% | 70-80% | >80% | Liquidate |
| **Liquidation Risk** | <5% | 5-10% | >10% | Emergency halt |

---

## Deployment Checklist

```
PHASE 6A: Position Risk (Week 1)
- [ ] VaR calculator functional
- [ ] Greeks estimation working
- [ ] 10 position test passing

PHASE 6B: Portfolio Aggregation (Week 2)
- [ ] Aggregator working on 50+ positions
- [ ] Correlation matrix calculated
- [ ] Concentration index computed

PHASE 6C: Dynamic Limits (Week 3)
- [ ] Limit manager responsive to volatility
- [ ] Bracket-adaptive limits working
- [ ] Stress level calculation accurate

PHASE 6D: Margin & Stress Testing (Week 4)
- [ ] Margin requirements correct
- [ ] Stress tests matching historical data
- [ ] Liquidation risk quantified

PHASE 6E: Dashboard & Alerts (Week 5)
- [ ] Real-time metrics display
- [ ] Alert engine generating correctly
- [ ] Integrating with Slack/Email

PHASE 6F: Integration & Testing (Week 6)
- [ ] Trading coordinator using risk gates
- [ ] End-to-end trade gating working
- [ ] 50+ integration tests passing

PHASE 6G: Performance & Hardening (Week 7)
- [ ] All calculations < 100ms
- [ ] Load test with 1000+ positions
- [ ] Production readiness review
```

---

## Success Metrics

### By Week 3
- ✅ Position risk calculated for all open positions
- ✅ Portfolio aggregation working
- ✅ Dynamic limits responsive

### By Week 6
- ✅ Trading coordinator respects risk limits
- ✅ Dashboard shows live risk metrics
- ✅ Alerts triggering correctly

### At Launch
- ✅ Zero margin calls (all prevented pre-trade)
- ✅ Portfolio VaR < 15% maintained
- ✅ All stress tests passed
- ✅ Performance targets met (< 100ms)

---

## Common Pitfalls to Avoid

1. **VaR Underestimation**
   - Use 99% confidence for stress, 95% for normal
   - Account for correlation breakdown in crashes

2. **Concentration Blindness**
   - Track by symbol AND by bracket
   - Monitor correlation-weighted concentration

3. **Margin Miscalculation**
   - Include both position AND leverage margin
   - Account for interest rates/funding costs

4. **Stress Test Overconfidence**
   - Black swan scenarios exceed historical range
   - Test with 50%+ moves (not just -20%)

5. **Dashboard Staleness**
   - Update every 1 second minimum
   - Queue alerts if they happen offline

---

## Phase 6 Success = Production Ready 🚀

Once Phase 6 is complete:
- ✅ Robust position sizing (Phases 1-5)
- ✅ Advanced risk management (Phase 6)
- ✅ Real-time monitoring & alerts
- ✅ Stress tested & resilient
- ✅ Ready for production deployment

**Next: Phase 7 (ML-based risk models, network analysis, compliance automation)**
