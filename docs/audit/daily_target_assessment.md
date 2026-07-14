# Daily Target Assessment — Five Qualified, Net-Profitable, Compoundable Trades/Day

Date: 2026-07-14. Scope: direct answer to "is this native system currently capable of
producing five qualified, net-profitable, compoundable trades per trading day?"

This document does not re-derive evidence — it synthesizes `daily_trade_funnel.md`,
`trade_lifecycle_map.md`, `strategy_contribution.md`, `profit_and_compounding_assessment.md`,
`current_state_assessment.md`, `component_inventory.md`, and `feature_ignition_matrix.md`,
plus the two backtest memory files (`edge-verdict-no-edge.md`,
`funding-carry-edge-candidate.md`). No new runtime session was executed for this document.

**Note on a missing input.** The brief lists
`docs/audit/runtime_observation_paper_session.md` as an upstream input. That file does
not exist in this pass — `daily_trade_funnel.md` (§"Source of counts") records that a
fresh paper-trade observation was explicitly skipped this session as unsafe (Binance
`exchange_client.py`'s mutating/trade-history calls have no `paper_mode` gate — see that
file for detail). This assessment is therefore built entirely on the prior dry-run
session (`runtime_timeline.md`) plus static code evidence and the live-process check in
`strategy_contribution.md` (a real `main.py` process was observed running today,
producing signals but zero real fills). This is flagged, not silently patched over.

---

## 1. Direct answer

**No. Zero net-profitable, compoundable trades per day, today, and not by a small
margin.** The system produced 0 real trade executions in every observation on record
this audit (the dry-run session, and today's live-process check which showed the
process running and generating 15 signals/cycle but 0 real fills — only 4 hardcoded
startup self-test log lines that are not real trades). The one full day of real trade
history that exists anywhere in this repo's journals (`trade_ledger.jsonl`,
2026-06-13, a month before today) produced 9 trades, of which only **1 was net-profitable
after fees** (`net_win_rate: 0.1111`, `cum_net_pnl_usdt: -0.5377`). Even on the one day
this system is known to have actually traded, it did not hit "5 net-profitable" — it
produced 1.

## 2. Current daily capability, broken out

### 2a. Profitable-trade capability: effectively zero, and not a volume problem

The dominant, best-evidenced cause is that the only strategy wired into the live
decision path (the legacy ML forecaster, bridged via `legacy_signal_adapter.py`) has a
**proven negative net edge**: −0.27%/trade backtest (3,140 samples, 38 models), −0.033
USDT/trade live (69 round-trips), confirmed on both 5m and 1h timeframes, with
confidence **mildly anti-predictive** (the high-confidence half performs worse than the
low-confidence half). This is not a hypothesis under test in this repo — it is a closed,
proven finding (`edge-verdict-no-edge.md`), independently corroborated by this session's
own dry-run backtest-calibration log line (confidence buckets 0.65-0.85 show 0-12.2%
historical win rate). Running this signal source more often, faster, or through more
symbols does not create profitable trades; it accelerates a negative-expectancy process.
This is why `daily_trade_funnel.md` found the funnel dies almost entirely at one gate
(`ConfFloor=0.9500`) — that gate is a defensive response to this proven no-edge finding,
not an accidental bottleneck.

### 2b. Compoundable-trade capability: zero, and separately blocked even if #1 were fixed

Even bracketing the strategy problem, `profit_and_compounding_assessment.md` establishes
that the system **cannot currently compute or persist a trustworthy net-of-fee realized
P&L per trade** under its default runtime configuration:

- The only fee-aware, currency-correct realized-P&L calculation
  (`fill_tracker.py::_handle_sell_fill`) lives on a component that is disabled by default
  (`polling_enabled=True` routes fill handling to `polling_coordinator.py`, which does
  not replicate the calculation).
- `shared_state.metrics["realized_pnl"]` — the field every downstream profit-reactive
  component reads (`NAVAttributionEngine`, `ObjectiveFeedbackController`,
  `AdaptiveCapitalEngine`) — is permanently `0.0` under default config as a direct
  consequence.
- The one per-trade P&L number that is computed and logged (`executor.py`'s
  `TRADE_CLOSED gross_pnl`) is explicitly gross, not net, and is never persisted anywhere
  queryable (log line only).
- **CORRECTED 2026-07-14**: `nav_protection.py` is actually wired via `main.py:412-433`
  (a directory-scoped grep in the original pass missed this — see
  `profit_and_compounding_assessment.md` §5). It is live and reactive to real NAV state.
  The remaining real gap is narrower: its `locked_profit_usdt`/`protection_floor_usdt`
  outputs are not yet *connected* to `daily_compounding.py`'s compounding formula (see
  remediation item #17) — a "connect two working systems" task, not "wire up dead code."
- `daily_compounding.py`'s flat-rollover NAV-freeze is a reasonable, structurally sound
  substitute for "don't compound on unrealized gains," but it is not the same guarantee
  as `compoundable_profit = max(0, reconciled_realized_net_profit − protected_profit_reserve)`
  — it sizes off raw exchange-reported NAV, gated only by portfolio-flatness, with no
  reconciled-realized-P&L or profit-reserve input.

Conclusion: **even if the strategy problem in §2a were solved tomorrow**, the system
would still need the fee-tracking and NAV-protection wiring fixes in
`profit_and_compounding_assessment.md` §10 before a "5 compoundable trades/day" claim
could be trusted rather than merely asserted.

### 2c. Ranked cause attribution (evidence-strength order, consistent with `daily_trade_funnel.md`)

1. **Strategy (no edge) — STRONGEST.** Proven across 3,140 backtest samples + 69 live
   trades, 5m and 1h. This is the root cause of near-zero trade volume (the ConfFloor
   gate exists because of it) and the root cause of the one real trading day's 11.1%
   net win rate. No gate-tuning, timeframe change, or execution-quality fix addresses
   this — it requires a different signal thesis.
2. **PnL-correctness / wiring (fee tracking, NAV↔compounding connection) — SECOND
   STRONGEST, independently proven, orthogonal to #1.** `fill_tracker` disabled-by-default
   (fixed 2026-07-14, see `changes_made.md`), `position_hydration_engine.py`'s
   commissionAsset mis-summation (fixed 2026-07-14). `nav_protection.py` itself is
   wired and live (corrected finding — was wrongly reported as unwired earlier this
   audit), but its outputs are not yet connected to `daily_compounding.py`'s
   compounding formula (remediation item #17, still open). These gaps made
   "net-profitable" and "compoundable" unverifiable even on the rare trade that clears
   the strategy gate — the fee-tracking half of this is now fixed; the NAV-to-compounding
   connection remains open.
3. **Wiring/timing (PERSIST_GATE vs. observation window) — real but sample-specific,
   not a steady-state daily cause.** Explains why the one 90-second dry-run sample saw
   0 candidates advance past the confirmation streak; does not explain a sustained
   daily zero, since a real day has ~288 five-minute bars for the streak to advance.
4. **Data (cold-start symbol discovery lag, WS reconnect churn) — minor multiplier,**
   not evidenced to persist past warm-up.
5. **Risk/capital/execution/exit stages (6-14 in `trade_lifecycle_map.md`) — NOT
   EVIDENCED as contributing, because none were ever runtime-exercised.** No candidate
   has reached them in any observed session. Treating them as "clean" would overstate
   what has been verified; treating them as "broken" would be speculation. One
   concrete, already-flagged defect exists in this range regardless of throughput
   (hardcoded `_compute_volatility_pct` placeholder in `capital_allocator.py`, and a
   possible `pnl_after_fees_usdt` telemetry-key contract mismatch flagged but not
   confirmed in `trade_lifecycle_map.md` Stage 13) — these should be fixed on
   general-correctness grounds independent of the strategy question.
6. **Market conditions (DOWNTREND/CASH_HEAVY/Fear&Greed=28 at last observation) —
   WEAKEST, plausible but unfalsifiable from one short sample.** Regime gates were
   never exercised because arbitration itself was never reached this session.
7. **System-wiring failures that were already found and fixed (position hydration
   `get_all_orders`) — resolved, not a current blocker,** but noted because it shows
   the audit process is finding real defects, and this class of defect (silent
   fallback masking missing state) is worth continuing to hunt for elsewhere (e.g. the
   `implementations.py:706-710` "arbitration engine `None` → default-pass" fallback,
   flagged but not observed to have fired).

## 3. Target progress state(s) that apply right now

Multiple states apply simultaneously to different parts of the system, and conflating
them would misdiagnose the fix. Evaluated against the enumerated vocabulary:

| State | Applies? | Evidence |
|---|---|---|
| `NOT_STARTED` | No | The system is actively running (`ps aux` shows a live `main.py` process today) and has one historical real trading day on record (2026-06-13). |
| `MONITORING` | **Yes, for the live strategy path** | The ML-forecaster path is continuously generating and scoring signals (15/cycle observed today) — the system is actively watching the market, it is simply not clearing its own quality bar. |
| `OPPORTUNITIES_AVAILABLE` | No, not currently evidenced | No signal has cleared PERSIST_GATE+ConfFloor in any observed session; the funding-carry backtest shows opportunities exist for a *different* strategy, but that strategy is not running. |
| `NO_QUALIFIED_OPPORTUNITIES` | **Yes, for the wired ML strategy** | 1,694 of ~1,750 recent BUY-decision log lines are `allowed=False`, dominated by `gate_2_confidence,gate_3_regime,gate_11_symbol_downtrend`. This is the correct, evidence-supported classification for the live path today — not `EXECUTION_BLOCKED` (nothing is trying to execute; nothing qualifies) and not `RISK_FROZEN` (no circuit breaker has tripped). |
| `EXECUTION_BLOCKED` | **Partially, for a structurally different reason** | `--mode=dry-run` and `--mode=paper-trade`'s missing paper-mode gating mean execution is either deliberately disabled (dry-run) or unsafe to test (paper-trade). This is a *test-harness* blocker, distinct from the strategy-side `NO_QUALIFIED_OPPORTUNITIES` state — a `--mode=live` run with a qualifying signal has never been observed to reach the executor at all, so whether real execution would succeed is unverified, not confirmed-blocked. |
| `RISK_FROZEN` | No | No daily-loss-limit or circuit-breaker trip evidenced in any session. |
| `DATA_DEGRADED` | No, transient only | Symbol-discovery cold-start lag observed once, resolved within one cycle; no sustained data degradation evidenced. |
| `PARTIAL_PROGRESS` | **Yes, in the profit/compounding sense** | Daily compounding logic is structurally sound for its narrow purpose (never compounds unrealized gains); TP/SL sizing and force-exit-with-fee-floor logic is well-designed. Real infrastructure exists; it has just never processed a real, net-profitable trade to compound. |
| `TARGET_ACHIEVED` | No | 0 net-profitable trades in any session this audit observed; 1-of-9 on the one historical real trading day, not 5. |
| `TARGET_MISSED_MARKET_CONDITIONS` | No, not the dominant cause per evidence | Regime/Fear&Greed data is suggestive but never independently exercised at runtime (arbitration never reached in the observed session); do not attribute the miss primarily to market conditions without stronger evidence. |
| `TARGET_MISSED_SYSTEM_FAILURE` | **Partially, for PnL-tracking correctness, not for order flow** | The fee/PnL tracking and NAV-protection wiring gaps in `profit_and_compounding_assessment.md` are genuine system-wiring failures that would prevent a trustworthy "net-profitable"/"compoundable" verdict even on a trade that otherwise succeeds. This is a real, evidenced contributor — narrower in scope than "the whole system is broken," but real. |
| `TARGET_MISSED_STRATEGY_PERFORMANCE` | **Yes — the dominant, best-evidenced state for the live path** | Proven negative edge (3,140 backtest samples + 69 live trades), independently corroborated by this session's own confidence-bucket win-rate check. This is the primary reason the target is missed, and no amount of downstream fixing changes it. |

**Composite state (most accurate single characterization):** the live strategy path is
in `MONITORING` + `NO_QUALIFIED_OPPORTUNITIES` (correctly, defensively) resulting in
`TARGET_MISSED_STRATEGY_PERFORMANCE`, with a secondary, orthogonal
`TARGET_MISSED_SYSTEM_FAILURE` contribution from unwired PnL/NAV-protection tracking
that would need fixing even after the strategy problem is solved. There is no evidence
supporting `RISK_FROZEN`, `DATA_DEGRADED` (sustained), or
`TARGET_MISSED_MARKET_CONDITIONS` as primary drivers today.

## 4. Does a dedicated daily-target monitor/controller exist?

**Built 2026-07-14, as part of this remediation pass** (`core_engine/native/daily_target_monitor.py::NativeDailyTargetMonitor`). At the time this document was first written, the answer was
"No — confirmed absent, consistent with the existing audit": no component in
`core_engine/native/` tracked "trades today," "net-profitable trades today," or "progress
toward N/day" as a first-class object; `daily_loss_limit_pct` (2%) remained the only
daily-scoped control, and it is a downside circuit breaker, not a progress/target tracker.
See `remediation_plan.md` item #18 for the implementation, wiring points, and test
coverage. It implements every requirement listed below, built only after items #3/#4
(net-of-fee PnL correctness) closed, per this document's own sequencing note.

**If one were built** (not implemented in this pass, per the brief), it should track, at
minimum:
- Per-UTC-day counters for each of the 14 lifecycle stages in `trade_lifecycle_map.md`
  (candidates generated → qualified → arbitrated → allowed → executed → filled →
  exited-profitably → compounded), so a funnel snapshot like
  `daily_trade_funnel.md` can be produced automatically instead of by manual log
  archaeology.
- A **reconciled, net-of-fee** realized P&L per closed trade (blocked today on the
  `fill_tracker`/`polling_coordinator` gap in §2b — the monitor cannot be built
  correctly until that gap closes, since a monitor built on the current gross/dead
  metrics would just automate a wrong answer).
- A running comparison against the 5/day target with the explicit progress-state
  vocabulary in §3, refreshed at least once per cycle, persisted so a restart doesn't
  lose same-day progress.
- An explicit distinction, surfaced in its own output, between "no qualified signal
  today" (strategy-side) and "signal qualified but blocked/failed downstream"
  (wiring-side) — exactly the split this document and `remediation_plan.md` insist on
  keeping separate, so a future on-call human reading the monitor's output isn't tempted
  to "fix" the wrong layer.
- A hard, code-enforced rule that the monitor is read-only with respect to gates — it
  must never auto-loosen ConfFloor, PERSIST_GATE, or any risk gate to chase the count.
  See §6.

## 5. Wiring-broken vs. no-edge — explicit separation

These two classes of problem require different owners, different timelines, and must
not be merged into a single "fix the bot" backlog item:

**A. System-wiring is broken (native/execution team, days-to-weeks, no new research
needed):**
- ~~`fill_tracker.py` disabled by default; its P&L calculation not replicated in
  `polling_coordinator.py`.~~ **Fixed 2026-07-14** — see `changes_made.md`.
- ~~`nav_protection.py` fully unwired (zero call sites in production code).~~
  **CORRECTED 2026-07-14 — this was wrong; it's wired via `main.py:412-433`.** See
  `profit_and_compounding_assessment.md` §5. The real remaining gap is narrower:
  connecting its outputs to `daily_compounding.py` (remediation item #17).
- ~~`position_hydration_engine.py` mis-sums fees across `commissionAsset`
  boundaries.~~ **Fixed 2026-07-14.**
- ~~No canonical `compute_net_trade_pnl()` function; `TRADE_CLOSED` log is gross-only and
  not persisted.~~ **Fixed 2026-07-14.**
- `capital_allocator.py`'s hardcoded volatility placeholder (`_compute_volatility_pct`
  returns 0.008 always).
- Possible `pnl_after_fees_usdt` telemetry-key mismatch between `make_sell_decision` and
  `main.py`'s consumption of `decision.telemetry` (flagged, unconfirmed).
- `config_loader.py`'s dead-but-plausible-looking env vars (operator-trap risk).
- No `--mode=paper-trade` gating on `exchange_client.py`'s mutating calls (safety gap,
  not a profitability gap, but a wiring defect nonetheless).
- Absence of a daily-target monitor/controller (§4) — pure wiring/observability gap,
  does not require new research to build once the PnL-correctness prerequisite above is
  met.

None of the above requires new signal research. All are addressable by the native team
with the existing strategy left exactly as-is.

**B. The wired strategy has no edge (requires new signal research, different owner,
weeks-to-months, no code-fix shortcut exists):**
- The legacy ML forecaster (`agents/ml_forecaster.py`), the only signal source in the
  live decision path, has a proven negative/anti-predictive edge across every timeframe
  and confidence bucket tested. This is closed, not exploratory — the finding is that
  **no execution, gate, fee, or model-selection change creates edge; the signal itself
  is the problem** (`edge-verdict-no-edge.md`).
- The only strategy in this codebase with a *proven positive* backtest edge
  (funding-carry, +1.22%/trade, 80% win, 361 spot-hedgeable perps) is not wired into the
  native runtime at all, and its own forward-proof daemon is currently stopped, stalled
  at 2 of the required ≥30 trades (`strategy_contribution.md`). Getting it live-ready
  requires (1) resuming/keeping the forward-proof daemon running to accumulate the
  remaining ~28 trades, (2) confirming the ≥30-trade gate nets positive, (3) integrating
  it into the supervised runtime — and even then, by its own author's characterization,
  it is a sparse, event-driven strategy that cannot alone guarantee a 5-trade/day floor.
- `statarb_discover.py` has no on-disk backtest artifact in this repo to independently
  verify its "tested & DEAD" status; if it is to be reconsidered, it needs its own
  documented backtest (new research effort) before any wiring decision is meaningful.

**Do not conflate.** Fixing every item in bucket A produces a system that correctly
executes, tracks, and compounds trades — but if the wired strategy is still the ML
forecaster, it will correctly execute, track, and compound a **negative-expectancy**
process, faster and more auditable. Bucket A is necessary but not sufficient. Bucket B
is necessary and, per the proven evidence, currently has no code-only shortcut.

## 6. Top-ranked blockers to reaching the target, in priority order

1. **No positive-edge signal source is wired into the live decision path.** (Bucket B)
   This is the dominant blocker. Fixing every other item on this list without
   addressing this one caps the achievable outcome at "efficiently, verifiably,
   compoundingly executing a proven-losing strategy."
2. ~~Net-of-fee realized P&L is not computed or persisted under default runtime
   config~~ — **fixed 2026-07-14** (Bucket A).
3. ~~`nav_protection.py` is fully unwired~~ — **corrected 2026-07-14: it was already
   wired.** The narrower remaining item is connecting its output to
   `daily_compounding.py`'s reserve term (remediation item #17, still open). (Bucket A)
4. **The funding-carry strategy (the only proven-positive edge in this codebase) is not
   wired into the native runtime, and its own forward-proof daemon is stopped**,
   stalled at 2/30 trades — the fastest available path to *some* real positive-edge
   volume is currently not even accumulating evidence toward its own readiness gate.
   (Bucket B, but the lowest-effort item in bucket B since execution is already
   testnet-validated)
5. **No daily-target monitor/controller exists**, so even if 1-4 were fixed, there is
   no automated, persistent tracking of progress toward "5/day" — today this audit had
   to reconstruct funnel counts by hand from logs. (Bucket A)
6. **PERSIST_GATE's 2-bar confirmation requirement was never observed to complete in
   the one short dry-run session**, but this is a sample-duration artifact, not a
   steady-state blocker — deprioritized versus 1-5 above.
7. **Several stages (arbitration through compounding, stages 6-14) have never been
   runtime-exercised at all**, so their correctness under real trade volume is
   genuinely unknown — not urgent relative to 1-4, but should be verified via a real
   (safely-gated) execution test before relying on them for a live-capital decision.

## 7. Safety constraint on these recommendations (binding on this document itself)

Per the brief: **do not recommend loosening confidence thresholds, forcing trade volume,
or tuning any gate to hit the 5/day count.** Every recommendation in §5/§6 above and in
`remediation_plan.md` is scoped to (a) correctness/observability of PnL and reconciliation,
(b) wiring an *already separately-validated-positive* strategy (funding-carry) into the
runtime, and (c) building monitoring — never to relaxing `ConfFloor`, `PERSIST_GATE`,
`gate_2_confidence`/`gate_3_regime`/`gate_11_symbol_downtrend`, or any risk gate on the
existing ML-forecaster path to manufacture volume. The 0.9500 confidence floor and the
downstream risk gates are, per the evidence in this document, functioning exactly as a
capital-preservation mechanism should against a proven negative-edge signal; loosening
them would not create edge, it would accelerate realized losses. Any future proposal to
loosen a gate must be backed by new evidence that the underlying signal's edge has
changed (e.g., a retrained/replaced model with its own independently-run backtest showing
positive net expectancy) — not by a desire to hit a trade-count target.
