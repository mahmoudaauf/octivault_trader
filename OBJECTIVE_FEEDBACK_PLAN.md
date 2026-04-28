# 🎯 Objective Feedback & Auto-Calibration Plan

**Objective contract:**
> Compound **+2%/day on NAV** via autonomous, risk-bounded, fee-aware multi-agent
> trading — measured continuously via checkpoints, protected by kill-switches,
> and reinvested at 50%.

This plan defines the **closed-loop control system** that makes the bot
*automatically calibrate itself* toward that objective.

---

## 1. The control problem

Translate the objective into a **set-point** the controller can track:

| Variable | Symbol | Set-point | Source |
|---|---|---|---|
| Daily NAV growth | `r_day` | **+2.00%** | `PROFIT_TARGET_DAILY_PCT` |
| Hourly pace (linearised) | `r_hour` | **+0.0833%** | `r_day / 24` |
| Per-cycle risk cap | `risk_c` | ≤ **0.5%** of NAV | `PROFIT_TARGET_MAX_RISK_PER_CYCLE` |
| Per-trade risk cap | `risk_t` | ≤ **2%** of NAV | `MAX_RISK_PER_TRADE` |
| Compound reinvest fraction | `α` | **0.50** | `PROFIT_TARGET_COMPOUND_THROTTLE` |
| Max drawdown (intra-day) | `dd_max` | ≤ **5%** | new (KILL_SWITCH_DD_PCT) |
| Min checkpoint cadence | `cp_min` | ≥ **1 / 15 min** | new (CHECKPOINT_HEARTBEAT_S) |

The controlled variables (knobs the loop is allowed to move):

| Knob | Range | Effect |
|---|---|---|
| `CONFIDENCE_FLOOR` | 0.50 – 0.85 | ↑ → fewer but stronger trades |
| `SIZE_MULTIPLIER` | 0.25 – 1.50 | scales `AdaptiveCapitalEngine` quote |
| `TARGET_THROUGHPUT_PER_HOUR` | 2 – 60 | desired trade rate |
| `MAX_OPEN_POSITIONS` | 1 – 10 | parallelism |
| `TP_BIAS_MULT` | 0.80 – 1.20 | tightens/loosens take-profit |

**Hard limits (never violated by the controller):**
`risk_t ≤ 2%`, `risk_c ≤ 0.5%`, `dd_max ≤ 5%`. These are *kill-switches*,
not optimisation targets.

---

## 2. Three-layer feedback loop

### L1 — Fast guard (every BUY attempt)
Already implemented in `core/profit_target_engine.py::check_global_compliance`.
**No change** — it gates trades against daily target & cycle risk.

### L2 — Medium feedback (every checkpoint, default 15 min) — **NEW**
`core/objective_feedback_controller.py` (this PR).

Each checkpoint:

1. **Measure** — read NAV, realized PnL, unrealized PnL, drawdown, trade count,
   win-rate, avg fee, slippage from `SharedState` + last `checkpoint_metrics.json`.
2. **Compute errors** vs. set-points:
   ```
   pace_error  = realized_pace_pct_per_h − 0.0833
   risk_error  = max(0, drawdown_pct − 5)
   thru_error  = trades_per_h − target_throughput_per_h
   ev_error    = avg_net_profit_per_trade − round_trip_cost
   ```
3. **Decide** with a bounded PI controller (proportional + integral):
   ```
   Δconfidence_floor = −Kp_c · pace_error − Ki_c · ∫pace_error
   Δsize_multiplier  = +Kp_s · pace_error · regime_gain
   Δthroughput_tgt   = +Kp_t · pace_error − Kp_d · risk_error
   ```
   All deltas clamped, then the new knob value is clamped to its allowed range.
4. **Act** — write knobs into `SharedState.runtime_overrides` (a hot-reload dict
   already read by `AdaptiveCapitalEngine`, `MetaController`, `Strategy`).
5. **Kill-switch** — if `risk_error > 0` for 2 consecutive checkpoints, or
   `dd > 5%`, set `SharedState.trading_halted = True` and emit alert.

### L3 — Slow calibrator (per session / per UTC day) — **NEW**
`core/objective_calibrator.py` (skeleton in this PR).

* Aggregates session outcome → `(achieved_r_day, dd, sharpe, win_rate, fee_drag)`.
* Updates EWMA priors over each knob's *effective range*
  (Bayesian-style with `α=0.3` smoothing).
* Writes recommended baseline values to `.env.calibrated` (loaded at next start).
* Demotes agents whose realised contribution-per-trade is < `0` over 3 sessions.

---

## 3. Telemetry contract (pre-condition for the loop to work)

The L2 loop is **only as good as the data feeding it**. The last 4-hour session
showed *0 checkpoints recorded* — the loop would have nothing to react to.
Therefore L2 *requires*:

| Field (in `checkpoint_metrics.json` per checkpoint) | Required |
|---|---|
| `nav` | ✅ |
| `realized_pnl_session` | ✅ |
| `unrealized_pnl` | ✅ |
| `drawdown_pct_from_peak` | ✅ |
| `trades_in_window` | ✅ |
| `win_rate_window` | ✅ |
| `avg_fee_bps`, `avg_slippage_bps` | ✅ |
| `agents_active`, `signals_per_min` | ✅ |

Acceptance gate: **the controller refuses to act** if any required field is
missing or older than `2 × CHECKPOINT_HEARTBEAT_S`. It logs a `STARVED`
warning instead of guessing.

---

## 4. Success gates (how we know the loop works)

| Gate | Metric | Threshold | Window |
|---|---|---|---|
| G1 Telemetry | checkpoints recorded | ≥ 1 / 15 min | per session |
| G2 Pace | rolling 4h NAV growth | ≥ +0.33% (= 4 × hourly) | rolling |
| G3 Daily | end-of-UTC-day NAV growth | ≥ +1.5% (75% of target) | per day |
| G4 Risk | intra-day max DD | ≤ 5% | per day |
| G5 Economics | avg net profit per trade | > round-trip cost | rolling 50 trades |
| G6 Convergence | knob changes per checkpoint | trending → 0 | over 24h |

The system is considered **on-objective** when G1–G5 are green for **3
consecutive UTC days** and G6 has converged (knobs stable within ±5%).

---

## 5. Iteration protocol

```
   ┌──────────────┐
   │ Run session  │
   └──────┬───────┘
          ▼
   ┌──────────────────┐   gates green for 3d?  ──► DONE
   │ Score G1..G6     │──┐
   └──────┬───────────┘  │ no
          ▼              ▼
   ┌──────────────────────────────────────┐
   │ Identify worst-failing gate          │
   │ G1 → fix telemetry plumbing          │
   │ G2 → loop will auto-adjust; observe  │
   │ G3 → expand size or throughput knob  │
   │ G4 → tighten risk knobs; lower size  │
   │ G5 → raise confidence floor / fees   │
   │ G6 → reduce Kp gains (oscillation)   │
   └──────────────────────────────────────┘
          │
          ▼
   apply fix → restart session → repeat
```

---

## 6. Files delivered with this plan

| File | Role |
|---|---|
| `OBJECTIVE_FEEDBACK_PLAN.md` | this document |
| `core/objective_feedback_controller.py` | L2 PI controller |
| `core/objective_calibrator.py` | L3 session-level learner (skeleton) |
| `objective_tracker.py` | CLI: scores G1–G6 from artefacts |
| `OBJECTIVE_DASHBOARD.md` | auto-updated status report |

---

## 7. Wiring (minimum integration to make it live)

In `🎯_MASTER_SYSTEM_ORCHESTRATOR.py::initialize_components`, after
`ProfitTargetEngine` is constructed:

```python
from core.objective_feedback_controller import ObjectiveFeedbackController
self.objective_fb = ObjectiveFeedbackController(
    config=self.config,
    shared_state=self.shared_state,
    profit_target_engine=self.profit_target_engine,
    logger=logger,
)
await self.objective_fb.start()   # spawns periodic task
```

That single block plugs the loop in. No other module needs to change because
the controller writes to `shared_state.runtime_overrides`, which existing
consumers already read.
