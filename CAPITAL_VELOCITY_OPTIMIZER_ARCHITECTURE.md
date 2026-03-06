# Capital Velocity Optimizer - Architecture Diagrams

Visual representations of how the optimizer fits into your system.

---

## System Architecture

### Overall Trading System
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │Discovery │    │MLForecaster│   │DerivedSignal│  │Portfolio│            │
│  │Agents    │    │Forecaster  │   │Generators │   │Authority│            │
│  └─────┬────┘    └─────┬──────┘   └─────┬────┘    └────┬─────┘            │
│        │               │               │             │                    │
│        │ Candidates    │ Signals       │ Signals     │ Velocity exits     │
│        └───────────────┴───────────────┴─────────────┴────────────┐       │
│                                                                    │       │
│                          ┌─────────────────────────────────┐      │       │
│                          │   CAPITAL VELOCITY OPTIMIZER    │◄─────┘       │
│                          │   (Recommendations Engine)      │              │
│                          └──────────────┬──────────────────┘              │
│                                        │                                │
│                          Rotation Recommendations                       │
│                                        │                                │
│  ┌─────────────────────────────────────▼──────────────────────┐        │
│  │          MetaController (Decision Orchestrator)           │        │
│  │  - Evaluates all signals + recommendations               │        │
│  │  - Applies governance gates (P9, capital floor)          │        │
│  │  - Routes to RotationAuthority & ExecutionManager        │        │
│  └──────────────────┬───────────────────────────────────────┘        │
│                     │                                                  │
│       ┌─────────────┴──────────────┬────────────────────┐            │
│       │                            │                    │            │
│       ▼                            ▼                    ▼            │
│  ┌──────────────┐      ┌──────────────────┐    ┌──────────────┐  │
│  │RotationAuth  │      │ExecutionManager  │    │TradeManager  │  │
│  │(Opportunity  │      │(Order Placement) │    │(Fills)       │  │
│  │Swaps)        │      │                  │    │              │  │
│  └──────────────┘      └──────────────────┘    └──────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Capital Velocity Optimizer - Internal Flow

```
optimize_capital_velocity()
    │
    ├─► measure_portfolio_velocity()
    │   ├─ For each position:
    │   │  └─ evaluate_position_velocity()
    │   │     └─ Calculate: velocity = (pnl / age) - fee
    │   └─ Aggregate: weighted average velocity
    │
    ├─► estimate_universe_opportunity()
    │   ├─ Read latest ML signals from SharedState
    │   └─ For each candidate:
    │      └─ estimate_opportunity_velocity()
    │         └─ Calculate: velocity = (confidence × move) / time
    │
    ├─► recommend_rotation()
    │   ├─ Identify recyclable positions (low velocity + aged)
    │   ├─ Compare vs best opportunities
    │   └─ Recommend rotation if gap > threshold
    │
    └─► Return VelocityOptimizationPlan
        ├─ portfolio_velocity_pct_per_hour
        ├─ opportunity_velocity_pct_per_hour
        ├─ velocity_gap
        ├─ rotations_recommended[]
        └─ analysis{...}
```

---

## Data Flow - Position Velocity Measurement

```
SharedState
    │
    ├─ Position[BTC]: {entry_time, unrealized_pnl_pct, value_usdt, ...}
    ├─ Position[ETH]: {entry_time, unrealized_pnl_pct, value_usdt, ...}
    └─ Position[SOL]: {entry_time, unrealized_pnl_pct, value_usdt, ...}
                  │
                  ▼
        evaluate_position_velocity()
                  │
        ┌─────────┼─────────┐
        │         │         │
    Calculate  Calculate   Calculate
      PnL/hr    Fee Cost   Net Velocity
        │         │         │
        └─────────┼─────────┘
                  │
                  ▼
        PositionVelocityMetric
        ├─ symbol: "BTC"
        ├─ pnl_per_hour: 0.0200 (2%/hr)
        ├─ holding_cost_bps: 10.0
        ├─ net_velocity: 0.0199 (1.99%/hr)
        └─ is_recyclable: false
```

---

## Data Flow - Opportunity Velocity Estimation

```
MLForecaster Signals
    │
    └─ Signal[SOL]: {
        confidence: 0.72,
        _expected_move_pct: 0.015,
        action: "BUY",
        ...
    }
            │
            ▼
    estimate_opportunity_velocity()
            │
    ┌───────┼──────────┐
    │       │          │
  Extract Multiply Divide by
  ML data  confidence time
            │       │       │
            └───────┼───────┘
                    │
                    ▼
    OpportunityVelocityMetric
    ├─ symbol: "SOL"
    ├─ ml_confidence: 0.72
    ├─ expected_move_pct: 0.015 (1.5%)
    ├─ expected_return_pct: 0.0108 (1.08%)
    └─ estimated_velocity_pct: 0.0108 (1.08%/hr)
```

---

## Decision Logic - When To Rotate

```
For each position in portfolio:
    │
    ├─ Is position old enough?  (age >= VELOCITY_MIN_POSITION_AGE_HOURS)
    │  ├─ NO  → Keep position
    │  └─ YES ▼
    │
    ├─ Is position velocity low or negative?
    │  ├─ NO  → Keep position
    │  └─ YES ▼
    │
    ├─ Is there a better opportunity available?
    │  └─ Best opportunity velocity > position velocity? 
    │
    ├─ Is velocity gap significant?
    │  └─ gap > VELOCITY_GAP_THRESHOLD_PCT?
    │
    └─ Output: rotation_recommendation
            {
                "exit_symbol": "BTC",
                "opportunity_symbol": "SOL",
                "velocity_gap_pct_per_hour": 0.98,
                "reason": "VELOCITY_OPTIMIZATION_GAP",
            }
```

---

## Velocity Comparison Matrix

### Example: 3-Position Portfolio

```
┌──────┬──────────────────────┬──────────────────┬─────────────┐
│ Pos  │ Age    │ PnL%  │ Vel  │ Recyclable? │ Keep?   │
├──────┼────────┼───────┼──────┼─────────────┼─────────┤
│ BTC  │ 2.5h   │ +0.1% │ 0.04%│ NO (low)    │ HOLD    │
│ ETH  │ 0.5h   │ +0.5% │ 1.0% │ NO (young)  │ HOLD    │
│ SOL  │ 30min  │ -0.1% │-0.2% │ NO (young)  │ HOLD    │
└──────┴────────┴───────┴──────┴─────────────┴─────────┘

Candidates:
┌──────┬───────┬──────┬──────────────────┐
│ Sym  │ Conf  │ Move │ Estimated Vel    │
├──────┼───────┼──────┼──────────────────┤
│ LINK │ 0.68  │ 1.0% │ 0.68%/hr         │
│ AVAX │ 0.71  │ 1.2% │ 0.85%/hr         │
│ NEAR │ 0.75  │ 1.5% │ 1.13%/hr         │
└──────┴───────┴──────┴──────────────────┘

Result:
  BTC (0.04%/hr) → NEAR (1.13%/hr)  Gap: 1.09%/hr ✓ Recommend
```

---

## Integration Points in MetaController

```
MetaController.__init__()
    │
    ├─► Initialize PortfolioAuthority
    ├─► Initialize RotationAuthority
    ├─► Initialize SymbolRotationManager
    ├─► ... other components ...
    │
    └─► Initialize CapitalVelocityOptimizer ◄─── ADD HERE
            └─ self.capital_velocity_optimizer = CapitalVelocityOptimizer(...)


MetaController.orchestrate()
    │
    ├─► _check_capital_floor_central()
    ├─► _gather_mode_metrics()
    ├─► _build_decisions(accepted_symbols_set)
    │
    ├─► ... collect ML signals, portfolio authority exits ...
    │
    └─► Call Capital Velocity Optimizer ◄─── ADD HERE
        │
        └─► velocity_plan = await optimize_capital_velocity(
                owned_positions=...,
                candidate_symbols=...
            )
            │
            └─► Log metrics + optionally use recommendations
```

---

## Recommendation Flow Through MetaController

```
                 velocity_plan
                      │
                      ├─ portfolio_velocity_pct_per_hour
                      ├─ opportunity_velocity_pct_per_hour
                      ├─ velocity_gap
                      │
                      └─ rotations_recommended[0]
                           ├─ exit_symbol: "BTC"
                           ├─ opportunity_symbol: "SOL"
                           ├─ velocity_gap_pct_per_hour: 0.98
                           ├─ confidence: 0.72
                           └─ reason: "VELOCITY_OPTIMIZATION_GAP"
                                │
                                ▼
                        MetaController Decision Logic
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                Log metric   Gate on    (Optional)
                 for viz    confidence  Boost
                            & gap       rotation
                            threshold   priority
                    │           │           │
                    └───────────┼───────────┘
                                │
                                ▼
                        PortfolioAuthority/RotationAuthority
                        (Existing governors remain in control)
```

---

## Governance Hierarchy

```
Layer 1: Capital Floor (MetaController)
    └─ Prevents BUY if capital < floor

Layer 2: PortfolioAuthority (Velocity Governance)
    ├─ Exits underperforming positions
    └─ Recycles capital if run_rate < target

Layer 3: RotationAuthority (Opportunity Swaps)
    ├─ Swaps low-score for high-score
    └─ Respects bracket-based restrictions

Layer 4: Capital Velocity Optimizer (NEW - Coordination)
    ├─ Measures what we're getting (realized velocity)
    ├─ Estimates what we could get (opportunity velocity)
    ├─ Quantifies improvement (velocity gap)
    └─ Recommends (doesn't execute)

All layers are independent. Optimizer enhances situational awareness.
```

---

## Configuration Sensitivity

```
VELOCITY_GAP_THRESHOLD_PCT
    │
    ├─ 0.1%  → Very aggressive (rotate frequently)
    ├─ 0.5%  → Balanced (default, practical)
    ├─ 1.0%  → Conservative (rotate rarely)
    └─ 2.0%  → Very conservative (almost disabled)

VELOCITY_MIN_POSITION_AGE_HOURS
    │
    ├─ 0.10  → Allow rotation of 6-min-old trades (risky)
    ├─ 0.25  → Allow rotation of 15-min-old trades (default, practical)
    ├─ 0.50  → Allow rotation of 30-min-old trades (stable)
    └─ 1.00  → Allow rotation of 1h-old trades (very stable)

VELOCITY_CONFIDENCE_MIN
    │
    ├─ 0.45  → Use all ML signals (noisy)
    ├─ 0.55  → Use medium-confidence signals (default, balanced)
    ├─ 0.70  → Use only high-confidence signals (selective)
    └─ 0.85  → Use only very-high-confidence signals (restrictive)
```

---

## Example: Real Trading Scenario

### Initial State
```
Portfolio:
  BTC: Entry 2.5h ago, +0.1% PnL → velocity = 0.04%/hr (stagnant)
  ETH: Entry 1.0h ago, +0.5% PnL → velocity = 0.50%/hr (good)

Candidates from ML:
  SOL: confidence=0.72, expected_move=1.5% → opportunity = 1.08%/hr (excellent)
  LINK: confidence=0.68, expected_move=1.0% → opportunity = 0.68%/hr (good)
```

### Optimizer Analysis
```
Step 1: Measure Portfolio
  - BTC: velocity = 0.04%/hr (recyclable: yes, age OK)
  - ETH: velocity = 0.50%/hr (recyclable: no, positive)
  - Portfolio average: 0.27%/hr

Step 2: Estimate Opportunities
  - SOL: opportunity = 1.08%/hr (strong signal)
  - LINK: opportunity = 0.68%/hr (weak signal)

Step 3: Identify Rotations
  - BTC → SOL: gap = 1.08 - 0.04 = 1.04%/hr ✓ Recommend
  - ETH → SOL: gap = 1.08 - 0.50 = 0.58%/hr ✓ Recommend (borderline)

Step 4: Output Plan
  {
    portfolio_velocity: 0.27%/hr,
    opportunity_velocity: 1.08%/hr,
    velocity_gap: 0.81%/hr,
    rotations_recommended: [
      {exit: BTC, opportunity: SOL, gap: 1.04%, confidence: 0.72},
      {exit: ETH, opportunity: SOL, gap: 0.58%, confidence: 0.72}
    ]
  }
```

### MetaController Decision
```
Check recommendation 1 (BTC → SOL):
  - Gap > threshold (1.04% > 0.5%)? ✓ YES
  - ML confidence > gate (0.72 > 0.65%)? ✓ YES
  → Inform RotationAuthority of opportunity
  → Let existing governance make final decision

Check recommendation 2 (ETH → SOL):
  - Gap > threshold (0.58% > 0.5%)? ✓ YES
  - ML confidence > gate (0.72 > 0.65%)? ✓ YES
  → But ETH has positive velocity, so lower priority
  → PortfolioAuthority probably keeps ETH
```

### Outcome
```
Action taken:
  - Exit BTC (1.04%/hr opportunity)
  - Enter SOL (1.08%/hr forecast)
  - Keep ETH (0.50%/hr, still positive)

Result:
  Portfolio velocity improves from 0.27%/hr to ~0.70%/hr
  (Realizations depend on actual fills + execution)
```

---

## Time Dimension - Velocity Over Session

```
Time (min) │ BTC Vel  │ ETH Vel  │ SOL Signal │ Action
───────────┼──────────┼──────────┼────────────┼─────────────
0          │ 0.05     │ 0.50     │ 1.08       │ Initial state
10         │ -0.02    │ 0.45     │ 1.05       │ BTC stagnating
20         │ -0.10    │ 0.40     │ 1.10       │ BTC worsening
30         │ ---      │ 0.45     │ 1.12       │ ✓ Rotated to SOL
40         │ ---      │ 0.48     │ 0.95       │ SOL working
50         │ ---      │ 0.50     │ 0.88       │ Portfolio improving
60         │ ---      │ 0.52     │ 0.82       │ Decision point: exit SOL?
```

---

## Summary Diagram

```
                    Real Positions
                         │
                    ┌────┴────┐
                    │ Velocity │
                    │ Measure  │
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              │                    │
         Portfolio          Opportunity
         Velocity           Velocity
         (Realized)        (Forecasted)
              │                    │
              │  Compare (Gap)     │
              ▼                    ▼
         ┌────────────────────────────┐
         │ Capital Velocity Optimizer │
         └────────────────────────────┘
                      │
            ┌─────────┴─────────┐
            │                   │
       Velocity Gap      Rotation
       Assessment      Recommendations
            │                   │
            └─────────┬─────────┘
                      ▼
              MetaController
              (Uses recommendations,
               applies governance,
               controls execution)
```

That's the complete architecture! The optimizer measures what you're getting, estimates what you could get, and recommends how to improve—while existing governance authorities remain in control.
