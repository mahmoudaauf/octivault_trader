# 🎯 Objective Dashboard

_Generated: 2026-04-28T07:45:35.948275Z_

**Overall:** 🔴 OFF-OBJECTIVE  (0/6 gates green)

| Gate | Name | Metric | Threshold | Status | Fix hint |
|---|---|---|---|---|---|
| G1 | Telemetry cadence | 0.50 checkpoints/h | ≥ 4 | ❌ | Ensure 2HOUR_CHECKPOINT_SESSION emits at least every 15 min |
| G2 | 4h rolling NAV pace | n/a | ≥ +0.333% | ❌ | Loop will raise size_multiplier; check ENTRY logic if persistent |
| G3 | Session NAV change | n/a | ≥ +1.50% (75% of daily target) | ❌ | Increase throughput target or extend session length |
| G4 | Intra-session max drawdown | n/a | ≤ 5.0% | ❌ | Tighten size_multiplier; verify kill-switch fired when needed |
| G5 | Avg net profit per trade | n/a | > 5.0 bps | ❌ | Raise confidence_floor; revisit fee/slippage assumptions |
| G6 | Controller convergence | n/a | σ ≤ 0.05 (knobs settling) | ❌ | Lower OBJ_KP_* gains if oscillating |

## How to read this
* **G1–G2** are *prerequisites* — without telemetry & pace, the controller is blind.
* **G3–G5** are *objective metrics* — the actual +2%/day contract.
* **G6** is *stability* — confirms the auto-calibration is converging.

Run `python3 objective_tracker.py` after each session to refresh.