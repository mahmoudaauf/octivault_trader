# Trade Lifecycle Map — Market Opportunity to Compoundable Realized Profit

Built on top of `data_flow_map.md`, `current_state_assessment.md`, `component_inventory.md`, and
`runtime_timeline.md` — no wiring conclusion in those files is re-derived here, only applied to the
14-point "counts as a trade" / "compoundable trade" definitions. Where this document adds a stage those
files didn't cover in detail (fill confirmation, exit gating, reconciliation-to-compounding), it cites
the exact file:line read this session.

**Scope note on safety:** this pass did not run the process. A fresh runtime observation was attempted
and explicitly skipped as unsafe for `--mode=paper-trade` — `place_order`/`cancel_order`/`get_order`/
`get_my_trades` in `core_engine/native/exchange_client.py` have no `paper_mode`/`paper_key` gate at all
(only `get_account()` is simulated), so `paper-trade` would send real signed HTTPS calls to
`https://api.binance.com`. Every stage below is therefore graded on the same evidence basis as the
existing audit: static code evidence, plus the one prior `--mode=dry-run` session in `runtime_timeline.md`
(which never reached execution by design). Stages are explicitly labeled **[STATIC ONLY]** or
**[RUNTIME-CONFIRMED, dry-run session]** — no stage in this document has ever been runtime-confirmed
through a live fill, because dry-run always stops before Phase 4.

## Legend

- **Blocks flow** = if this stage fails/gates negative, no trade reaches later stages this cycle.
- **Loss/missed-opportunity source** = whether the stage's current behavior, as evidenced, tends to
  produce net losses, false negatives (killing viable trades), or false positives (letting bad trades
  through).

## Stage-by-stage map

### 1. Market discovery
- **Component/file**: `core_engine/native/symbol_discovery.py` (`NativeSymbolDiscovery`), wallet-scan
  based; `symbol_rotator.py` (`SymbolRotator`, TOP_N=8/2h) for rotation candidates.
- **Trigger**: per-cycle wallet REST scan (`main.py` cycle loop) + 2h rotation timer.
- **Input**: exchange wallet balances (non-zero holdings).
- **Output**: traded-symbol universe fed to `market_data_websocket` (re-subscribe) and `MLForecaster`.
- **Monitored**: yes, log line at discovery (`runtime_timeline.md` T+36s, T+48s).
- **Functioning per evidence**: **RUNTIME-CONFIRMED (dry-run)** but degraded — returned 0 symbols on
  first attempt, took a full extra cycle (~12s) to populate 5 real holdings. Every symbol-set change
  forces a full WS disconnect/reconnect (2 churns in a 54s/3-cycle window).
- **Blocks flow**: yes — if discovery returns empty, `MLForecaster` falls back to 10 hardcoded
  `DEFAULT_SYMBOLS`, not real holdings; a symbol not in that fallback list and not yet discovered gets
  zero signal generation that cycle.
- **Loss/missed-opportunity source**: **missed opportunity** — cold-start discovery lag (~1-2 cycles)
  plus WS reconnect churn on every discovery change is dead time with no signal generation for the
  newly-discovered symbols during the reconnect window.

### 2. Market data
- **Component/file**: `market_data.py` (REST poller), `market_data_websocket.py` (WS ticks + kline
  pre-fetch).
- **Trigger**: continuous (WS) + periodic REST (`price_refresh=60s` per `polling_coordinator.py:122`).
- **Input**: Binance market data endpoints.
- **Output**: `NativeSharedState` prices/klines, consumed by `MLForecaster`, `tp_sl_engine.py`,
  `symbol_rotator.py`.
- **Monitored**: yes — staleness threshold (`stale_threshold_sec`, 30s default per `data_flow_map.md`).
- **Functioning per evidence**: **RUNTIME-CONFIRMED** — 64 symbols priced at startup, kline pre-fetch (10
  symbols) completed in 8s, WS connected and receiving messages by T+19s.
- **Blocks flow**: yes if stale/missing — no evidence this fired in the observed session.
- **Loss/missed-opportunity source**: none observed this pass; WS reconnect churn (see Stage 1) is the
  only concretely observed cost, attributable to discovery, not market data itself.

### 3. Strategy analysis (MLForecaster)
- **Component/file**: `agents/ml_forecaster.py` (legacy), invoked via
  `core_engine/native/legacy_signal_adapter.py` → `signal_manager_bridge.py`.
- **Trigger**: `run_once()` per cycle (native decision cadence).
- **Input**: klines/price data, loaded `.keras` models per symbol (via `model_manager.py`).
- **Output**: per-symbol `{action, confidence}` (BUY/SELL/HOLD + probability).
- **Monitored**: yes — INFO-level per-symbol log lines confirmed in `runtime_timeline.md`.
- **Functioning per evidence**: **RUNTIME-CONFIRMED, but functioning per its own proven-negative
  calibration.** This is the single most consequential fact in this map: the memory file
  `edge-verdict-no-edge.md` (3,140 backtest samples + 69 live trades, 5m AND 1h) proved this model's
  confidence score has **no predictive relationship with net P&L**. The observed session's own internal
  backtest check corroborates this independently: confidence buckets 0.65-0.85 show 0-12.2% historical
  win rate (negative EV) — which is precisely why `ConfFloor` is set to 0.9500 (see Stage 5). The model
  is "working" in the sense of producing scores on schedule; it is not working in the sense of producing
  a signal that predicts profit. Every downstream gate (PERSIST_GATE, ConfFloor, arbitration, playbook
  confidence floors) is compensating for this, not for noise in an otherwise-valid signal.
- **Blocks flow**: no (produces candidates for the next stage regardless of quality).
- **Loss/missed-opportunity source**: **root-cause loss source for the whole system.** If confidence has
  no edge, every trade that does eventually clear all downstream gates is, per the proven evidence, not
  distinguishable from a coin-flip at best. This is a strategy defect, not a wiring defect — no amount of
  fixing gates below this stage repairs it. The only validated alternative in this codebase is the
  funding-rate carry strategy (`funding_carry_backtest.py`, +1.22%/trade, 80% win, 361 spot-hedgeable
  perps, backtest-only) and it is **not wired into this path at all** (see Stage 4 and
  `strategy_contribution.md`).

### 4. Signal (native cross-check + gating entry)
- **Component/file**: `core_engine/native/signals.py` (`NativeSignalEngine`, RSI/MACD/MA/Momentum,
  ±0.05 confidence nudge); `signal_manager_bridge.py` merges legacy ML output + this cross-check +
  (gated-off by default) `paper_signal_generator.py`.
- **Trigger**: same cycle as Stage 3, called from `SituationEngineImpl.get_all_signals()`.
- **Input**: MLForecaster output.
- **Output**: adjusted-confidence signal list.
- **Monitored**: **no** — `data_flow_map.md` already flags "not independently confirmed in log output
  (no per-indicator log lines observed at INFO level)".
- **Functioning per evidence**: **STATIC ONLY** for the cross-check math itself; wiring confirmed but the
  ±0.05 adjustment's effect was never independently observed in logs.
- **Blocks flow**: no (adjusts, doesn't gate).
- **Loss/missed-opportunity source**: none identified beyond the inherited Stage 3 problem; the ±0.05
  nudge cannot repair a signal proven to have zero edge, and could just as easily push a losing signal
  over a threshold as filter it out — no evidence either way, flagged as untested on a live path.

### 5. Qualification (PERSIST_GATE / ConfFloor)
- **Component/file**: inside `agents/ml_forecaster.py` (legacy), surfaced via the same bridge as Stage 3.
- **Trigger**: every signal candidate, every cycle.
- **Input**: raw signal + confidence, rolling confirmation streak state.
- **Output**: pass/hold verdict; only signals with streak=2/2 AND confidence ≥ 0.9500 pass.
- **Monitored**: yes — explicit log lines (`PERSIST_GATE ... streak=1/2`, `ConfBacktest: required=0.9500`).
- **Functioning per evidence**: **RUNTIME-CONFIRMED** — in the one observed session, the single
  candidate (BNBUSDT, conf=0.83) was held at streak=1/2 for all 3 cycles (never advanced — consistent
  with the 5-minute signal timeframe not producing a new closed bar in the ~54s window) and separately
  would have failed the 0.9500 floor even had the streak advanced.
- **Blocks flow**: **yes, this is the dominant real-world gate** — 100% of candidates in the observed
  session were blocked here, before arbitration ever saw them.
- **Loss/missed-opportunity source**: **defensive, not primarily a loss source** — the 0.9500 floor is a
  correct response to the proven no-edge finding (Stage 3), not an arbitrary blocker. It is, however,
  aggressive enough that it may be filtering out ~100% of signal volume, which is exactly why zero trades
  were observed reaching decision-making. This is the crux of the "daily target" tension addressed in
  `daily_target_assessment.md`: the gate that prevents losses is the same gate that prevents any trades
  at all under the current (no-edge) model.

### 6. Arbitration
- **Component/file**: `core_engine/native/arbitration_engine.py` (`NativeArbitrationEngine.evaluate()`),
  invoked from `core_engine/implementations.py::DecisionEngineImpl.evaluate_signal()`
  (`implementations.py:690-720`), called from both `make_buy_decision` (`:768`) and `make_sell_decision`
  (`:884`).
- **Trigger**: every signal that clears Stage 5 and reaches the decision phase.
- **Input**: symbol, signal_type, edge_score.
- **Output**: `{passed, gates_status, blocking_gates, reason}` dict; also feeds
  `get_symbol_quality(symbol)` used for `agent_quality` in the probability score.
- **Monitored**: partial — one-time wiring confirmation log line only (`✓ Wired DecisionEngine →
  arbitration_engine (L5)`); no per-evaluation log line confirmed either statically or at runtime.
- **Functioning per evidence**: **wired-by-code-evidence, confirmed live path** (this audit's own
  Section E correction in `current_state_assessment.md` — the earlier "dead code" finding was wrong), but
  **never runtime-exercised** — zero calls in the observed session because zero signals survived Stage 5.
  `implementations.py:706-710` has a defensive fallback: if `arbitration_engine` is `None`, it
  **defaults to `passed=True`** ("MVP default-pass") rather than blocking — a silent-fallback pattern
  worth flagging (same family as the position-hydration fallback already fixed this audit), though not
  observed to have fired (arbitration engine is confirmed instantiated).
- **Blocks flow**: yes, when reached — gates_status/blocking_gates can veto a BUY/SELL.
- **Loss/missed-opportunity source**: **unknown/unverified this pass** — genuinely untested at runtime on
  the live path; its cooldown/loss-streak/symbol-quality logic (`record_buy`, `record_sl_exit`,
  `record_win`, `record_loss`, `record_trade_outcome` — all called post-execution from `main.py:380-398`)
  has never fired either, since execution has never fired. Cannot confirm it works correctly under real
  trade volume; also cannot confirm it doesn't over-block.

### 7. Risk (playbook fits, probability score, confidence floor)
- **Component/file**: `core_engine/implementations.py::make_buy_decision` (`:730-803`), using
  `select_playbook()`, `compute_probability_score()` (`quant_reasoning.py`), plus
  `concentration_guard.py` / `regime_gate.py` (consumed inside `decisions.py`/`arbitration_engine.py` per
  `component_inventory.md` — untested, no dedicated test file).
- **Trigger**: same call as Stage 6, sequential within `make_buy_decision`/`make_sell_decision`.
- **Input**: market_regime, portfolio_state, edge_score, arbitration result.
- **Output**: `allowed` boolean, `blocked_reason`, `probability_score` vs. playbook `confidence_floor`
  (raised to ≥0.70 under `risk_state == DEFENSIVE`, `implementations.py:793-794`).
- **Monitored**: yes — `QUANT_LOOP_SUMMARY` log line observed (`market_regime=DOWNTREND
  portfolio_state=CASH_HEAVY allowed=False execution_result=NONE`, `runtime_timeline.md` T+43s).
- **Functioning per evidence**: **RUNTIME-CONFIRMED** at the summary-log level (fires every cycle
  regardless of whether a signal reached this far — it's evaluating the ambient situation, not gated by
  Stage 5). The specific per-decision path (`make_buy_decision` body) was **never runtime-exercised**
  since no signal reached it in the observed session.
- **Blocks flow**: yes — `allowed=False` on `BUY_BLOCKED_BY_PLAYBOOK`, `NO_EXECUTABLE_CAPITAL`, or
  `PROBABILITY_BELOW_FLOOR`.
- **Loss/missed-opportunity source**: none independently confirmed; layered on top of an
  already-degenerate signal (Stage 3), so any false-negative/false-positive behavior here is secondary to
  the root cause.

### 8. Affordability / Allocation
- **Component/file**: `core_engine/native/capital_allocator.py` (`NativeCapitalAllocator
.allocate_for_buy`, `:89-338`), `capital_policy.py::compute_spendable_quote`, `adaptive_capital_engine.py`
  (NAV≥$100 branch only), `daily_compounding.py` (`DailyCompoundingPolicy.sizing_nav()`, constructed
  internally at `capital_allocator.py:83-86`, called every `allocate_for_buy` at `:151`).
- **Trigger**: called from `make_buy_decision` (`implementations.py:780-787`), only if arbitration passed.
- **Input**: NAV, reserved_quote_total, quote_reserve_ratio (10% default), min-order-size per symbol.
- **Output**: `suggested_quote_usdt` (0 if unaffordable — blocks with `NO_EXECUTABLE_CAPITAL`).
- **Monitored**: yes — `[allocator] %s: nav=... mult=... spendable=... usdt=... qty=...` log line
  (`capital_allocator.py:337-338`).
- **Functioning per evidence**: **STATIC ONLY / wired-by-code-evidence** — never runtime-exercised (0
  BUY decisions reached this call in the observed session). One known defect already flagged by the
  prior audit: `_compute_volatility_pct` (`capital_allocator.py:395`) returns a **hardcoded 0.008
  placeholder**, not real rolling volatility — sizing under this path is not actually volatility-aware
  despite the code structure implying it is.
- **Blocks flow**: yes (`NO_EXECUTABLE_CAPITAL` when `suggested_quote_usdt <= 0`, or when min-order-size
  exceeds spendable capital, `:317-326`).
- **Loss/missed-opportunity source**: the hardcoded volatility placeholder is a real (if currently
  unexercised) risk-sizing defect — position sizes for volatile symbols are not actually scaled down
  relative to calm symbols, understating risk on the (unproven) occasions a trade does execute.

### 9. Execution
- **Component/file**: `core_engine/native/executor.py` (`NativeExecutor.execute`, `:102`), invoked via
  `engines.execution.execute_decision(decision)` from `main.py:364`, itself gated by `main.py:362`
  (`if mode != "dry-run"`).
- **Trigger**: once per allowed decision, only outside dry-run.
- **Input**: `TradeDecision` (symbol, action, quantity/suggested_quote_usdt).
- **Output**: order placed via `NativeOrderExecution` → `NativeExchangeClient.place_order()`
  (`exchange_client.py:397`).
- **Monitored**: yes by design (order logs), but **critically, this project's own risk-notice this
  session found `place_order`/`cancel_order`/`get_order`/`get_my_trades`
  (`exchange_client.py:397-484`) have zero `paper_mode`/`paper_key` gating** — they always build a real
  signed request to `https://api.binance.com` (`DEFAULT_BASE_URL`, `:61`) unless the separate `testnet`
  flag is set. Only `main.py:362`'s `mode != "dry-run"` check stands between a decision and a real
  order.
- **Functioning per evidence**: **NEVER RUNTIME-EXERCISED** in any session on record — the one observed
  dry-run session never entered this phase by design (`exe=0` on every cycle), and no `--mode=paper-trade`
  or `--mode=live` session was run this pass (explicitly judged unsafe, see scope note above).
- **Blocks flow**: `mode == "dry-run"` blocks it entirely (by design); otherwise nothing local blocks
  it beyond decision `allowed=True`.
- **Loss/missed-opportunity source**: **unverified, highest-uncertainty stage in the whole lifecycle.**
  There is no evidence in this codebase's history of a single confirmed real fill through this exact
  code path. All downstream stages (fill confirmation, TP/SL, exit, reconciliation) inherit this same
  "wired but never observed executing" status.

### 10. Fill (confirmation / position registration)
- **Component/file**: `NativeExecutor` itself performs "deliberately duplicated" fill/position
  registration bookkeeping (`executor.py:497, 690` comments) to compensate for `fill_tracker=None` under
  default `polling_enabled=True`; `NativePollingCoordinator`'s fills-reconciliation loop
  (`polling_coordinator.py:539-566`, 60s interval, `get_account_trades()` per symbol,
  `:643` "entry reconciled: ... (Binance avg fill, N trades)") is the actual periodic fill-truth source
  under default config.
- **Trigger**: immediately post-order (executor's own bookkeeping) + 60s periodic reconciliation pass.
- **Input**: order response / `get_my_trades` per-symbol trade history.
- **Output**: position registered in `NativeSharedState`, entry price reconciled to Binance's actual
  average fill (not the requested price).
- **Monitored**: yes — dedicated `[PollingCoordinator:FILLS]` log line.
- **Functioning per evidence**: **STATIC ONLY** — this exact reconciliation loop is the same
  `get_my_trades`-based mechanism whose pagination bug was fixed this session per recent commit history
  (`b84f1dd5 fix: paginate get_my_trades so hydration isn't truncated at 500 trades/symbol`), which is
  independent confirmation the code path is real and has had at least one real defect already found and
  fixed — but no live fill has been observed going through it this audit.
- **Blocks flow**: no (records after the fact); a broken fill-reconciliation would silently mis-record
  entry price/quantity rather than blocking anything.
- **Loss/missed-opportunity source**: mis-recorded entry price would corrupt every downstream P&L
  calculation (Stage 12-13) without necessarily producing an error — a silent-corruption risk class, not
  independently proven to have occurred, but structurally plausible given the intentional duplication
  the code comments themselves flag as a compensating workaround rather than a single source of truth.

### 11. Position management (TP/SL)
- **Component/file**: `core_engine/native/tp_sl_engine.py` (`NativeTPSLEngine`), Tier 2 fee-aware +
  time + trailing; `.check_triggers()` per position per cycle (`orchestrator.py:531` per prior audit,
  duplicated live in `main.py` around lines 296-310 for the actual production path — TP/SL signals
  injected as synthetic SELL signals, `_tpsl_signals`); `.recalculate_aged_positions()` every 300s
  (`main.py:436-446`).
- **Trigger**: every cycle (trigger check) + every 300s (aged-position recalculation).
- **Input**: live position marks vs. entry price, configured TP/SL/trailing/time parameters.
- **Output**: synthetic SELL signal with `tpsl_trigger` reason, prepended to `all_signals` so it takes
  priority in the decide phase (`main.py:308-310`).
- **Monitored**: yes — `[main:TPSL]` and `[main:TPSL-RECALC]` log lines.
- **Functioning per evidence**: **STATIC ONLY / wired-by-code-evidence** — `NativeTPSLEngine` is
  confirmed "Active" and "armed" in `current_state_assessment.md`, but no trigger fired in the observed
  session because there were zero open positions being tracked live (the 5 real positions found by the
  position-hydration fix were found in a separate check, not necessarily loaded into this same live
  session's `NativeSharedState` at that moment — worth re-verifying together, flagged as a follow-up).
- **Blocks flow**: no — it's an additional signal source, not a gate; but per `main.py:337-341`, TP/SL
  exits (`_is_tpsl`) bypass the `decision_due` cadence check, so they fire every cycle regardless.
- **Loss/missed-opportunity source**: `make_sell_decision`'s forced-exit floors (Stage 12 detail below)
  interact directly with this stage's triggers — a TIME_FORCE_EXIT can force a loss up to -100% (no
  floor) specifically to avoid "capital deadlock," per `implementations.py:987-989` comment referencing
  a prior "BIO -7.8% incident." This is a designed trade-off (avoid indefinite capital lock-up) but is
  explicitly a sanctioned loss-acceptance path, not a bug — worth flagging as a real, if intentional,
  source of realized losses that a naive "trade count" metric would still count as a completed
  round-trip.

### 12. Exit (SELL decision profitability gate)
- **Component/file**: `core_engine/implementations.py::make_sell_decision` (`:839-1030`, read in full
  this session).
- **Trigger**: every SELL-type signal (from Stage 3 or from Stage 11's synthetic TP/SL signal).
- **Input**: open position (`qty`, `entry_price`, current mark), fee_pct (derived from
  `fee_bps(shared_state, "taker") * 2 + exit_slippage_bps`, floored at 0.20%, `:932-953` — this is a real,
  specific improvement over an earlier hardcoded 0.002 fee-only estimate per the inline comment
  referencing a previously-booked "net-losing win" defect, e.g. HBAR/UNI sold +0.31% gross = -0.07% net).
- **Output**: SELL decision only if `pnl_after_fees > 0`, OR an explicitly sanctioned forced/protection
  exit within its own loss floor (`:989-990`: -10% floor for SL_HIT, -100%/no floor for
  TIME_FORCE_EXIT, -1% floor for NAV-protection-mode exits).
- **Monitored**: yes — explicit log lines for every branch (`⏱️ SELL ... accepting loss`, `🚫 SELL ...
  holding`, `🚫 SELL skipped ... unprofitable after fees`).
- **Functioning per evidence**: **STATIC ONLY** — logic is sound and specifically hardened against a
  previously-identified real defect (the HBAR/UNI net-loss "win" the inline comment documents), but never
  runtime-exercised (0 open positions with live SELL signals in the observed session).
- **Blocks flow**: yes — this is the core "only realize a net-positive trade" gate for organic SELL
  signals; forced/protection exits deliberately bypass it within their own floors.
- **Loss/missed-opportunity source**: **this is the correct place for the "net PnL positive" and "fees
  deducted"/"slippage included" requirements from the compoundable-trade definition to be enforced**, and
  the evidence shows it genuinely does so for organic exits. The intentional bypasses (forced/protection
  exits) are the one place the system knowingly realizes a loss by design — correctly labeled as such in
  logs, not hidden.

### 13. Realized PnL
- **Component/file**: no single dedicated "realized PnL ledger" component was found this pass beyond
  the `pnl_after_fees`/`pnl_after_fees_pct` computed inline in `make_sell_decision` and passed through
  `decision.telemetry`; `main.py:384-398` reads `telemetry.get("pnl_after_fees_usdt", 0.0)` post-execution
  to classify win/loss for arbitration feedback (`record_win`/`record_loss`/`record_trade_outcome`).
- **Trigger**: immediately after a successful SELL execution.
- **Input**: `decision.telemetry["pnl_after_fees_usdt"]`.
- **Output**: arbitration-engine outcome recording (Stage 6 feedback loop) + (presumably)
  `NativeTradeJournal` write (`trade_journal.py`, confirmed in bootstrap but its exact write-path for
  realized PnL was not independently re-traced this pass — flagged as a follow-up, not confirmed either
  way this session).
- **Monitored**: partially — arbitration feedback calls are logged as counters, not confirmed to log the
  raw PnL value itself in an easily auditable line.
- **Functioning per evidence**: **STATIC ONLY**, never runtime-exercised.
- **Blocks flow**: no (records after the fact).
- **Loss/missed-opportunity source**: none identified beyond inherited uncertainty; but note the
  telemetry key name (`pnl_after_fees_usdt`) is read at `main.py:385` from `decision.telemetry`, while
  `make_sell_decision`'s own local variable is `pnl_after_fees` (a float, not explicitly confirmed placed
  into `telemetry` under that exact key within the code read this session — the snippet read stopped at
  line ~1030; this is a **candidate contract mismatch worth flagging for verification**, not confirmed
  broken, since the full function body past line 1030 was not read in this pass).

### 14. Reconciliation
- **Component/file**: `NativePollingCoordinator`'s balance loop (`polling_coordinator.py:281-401`, 40s
  interval, "Synced balance from exchange"), position loop (`:315-347`, 25s), and fills-reconciliation
  loop (`:539-566`, 60s) — these three together reconcile `NativeSharedState` against exchange truth
  independent of what the in-process executor believes happened.
- **Trigger**: continuous, staggered per-loop intervals.
- **Input**: exchange REST responses (`get_account`, `get_my_trades`, presumably `get_open_orders`).
- **Output**: corrected balance/position/entry-price state in `NativeSharedState`.
- **Monitored**: yes — dedicated log lines per loop.
- **Functioning per evidence**: **RUNTIME-CONFIRMED for balance only** (T+18s, "$57.85 free/locked" in
  the observed session) — but the **position-recovery half of reconciliation was confirmed broken this
  audit** (`get_all_orders` AttributeError, now fixed per `changes_made.md` and the
  `get_my_trades`-pagination fix in commit `b84f1dd5`). Fills-reconciliation loop itself: **STATIC ONLY**,
  never observed processing a real fill.
- **Blocks flow**: no directly, but a reconciliation mismatch would silently corrupt the NAV/position
  state that Stage 8 (allocation) and compounding read from — see `profit_and_compounding_assessment.md`.
- **Loss/missed-opportunity source**: the now-fixed `get_all_orders` bug was, per this audit's own
  finding, **the highest-severity concrete defect found** — it would have silently dropped real open
  positions from `shared_state` on any restart during the period it was broken, i.e. reconciliation could
  have failed exactly the "reconciled against balances/portfolio state" requirement in the trade
  definition, for an unknown historical window predating this fix.

## Compoundable-capital stage (beyond the 14-point trade definition)

- **Component/file**: `DailyCompoundingPolicy`/`DailyCompoundingState` (`daily_compounding.py`,
  constructed inside `NativeCapitalAllocator.__init__`, `capital_allocator.py:83-86`), `sizing_nav()`
  called every `allocate_for_buy` (`:151`).
- **Trigger**: every BUY allocation call (Stage 8), reads current NAV to decide how much of realized
  gains are available for the next trade's sizing.
- **Input**: NAV (which itself depends on Stage 14 reconciliation being correct), daily compounding
  state (persisted).
- **Output**: a NAV figure used to size the next allocation — i.e. this is the literal mechanism by which
  "realized profit becomes available for controlled reinvestment," the compoundable-trade definition's
  core requirement.
- **Monitored**: indirectly, via the allocator's own summary log line.
- **Functioning per evidence**: **STATIC ONLY, never runtime-exercised** — like all of Stages 8-14, zero
  BUY/SELL cycles have completed in any observed session, so this policy has never actually compounded a
  real realized gain in this audit's history.
- **Blocks flow**: could effectively throttle allocation size if NAV is understated (e.g., due to a
  reconciliation gap), but no evidence of this occurring.
- **Loss/missed-opportunity source**: **entirely dependent on the correctness of every upstream stage**,
  most acutely Stage 14 (reconciliation) and Stage 3 (signal edge). Given Stage 3 is proven to have no
  edge, whatever capital does compound through this policy compounds trades that are — per the proven
  evidence — not distinguishable from noise, which is the deepest-rooted risk in the whole compounding
  design: it will compound losses just as mechanically as it would compound genuine edge, and currently
  has no genuine edge to compound.

## Summary table — wiring status vs. runtime status

| # | Stage | Wired (code evidence) | Runtime-exercised | Blocks flow | Loss/missed-opp risk |
|---|---|---|---|---|---|
| 1 | Market discovery | Yes | Yes (dry-run) | Yes (indirect via fallback) | Missed-opp (cold-start lag, WS churn) |
| 2 | Market data | Yes | Yes (dry-run) | Yes (staleness gate, unobserved) | None observed |
| 3 | Strategy analysis (MLForecaster) | Yes | Yes (dry-run) | No | **Root cause — proven no edge** |
| 4 | Signal cross-check | Yes | Unconfirmed (no per-indicator logs) | No | Unknown, inherits Stage 3 |
| 5 | Qualification (PERSIST_GATE/ConfFloor) | Yes | Yes (dry-run) | **Yes — dominant real gate** | Defensive (correct response to Stage 3) |
| 6 | Arbitration | Yes (corrected finding) | No | Yes, when reached | Unverified; silent default-pass-if-absent fallback exists |
| 7 | Risk (playbook/probability) | Yes | Partial (summary log only) | Yes | Unverified |
| 8 | Affordability/Allocation | Yes | No | Yes | Hardcoded volatility placeholder |
| 9 | Execution | Yes | **No — never observed, ever** | `dry-run` blocks; nothing else does | Highest uncertainty in system |
| 10 | Fill confirmation | Yes | No | No (records after fact) | Silent entry-price corruption risk (unproven) |
| 11 | Position mgmt (TP/SL) | Yes | No (0 live positions in session) | No (adds signals) | Intentional forced-loss floors (documented) |
| 12 | Exit (profitability gate) | Yes | No | Yes | Correctly hardened vs. a prior real defect |
| 13 | Realized PnL | Yes | No | No | Possible telemetry-key contract mismatch (unconfirmed) |
| 14 | Reconciliation | Yes | Partial (balance only) | No directly | Position-recovery half was broken; now fixed |
| — | Compoundable capital (DailyCompoundingPolicy) | Yes | No | No | Compounds whatever Stage 3 produces — currently no edge |

## Bottom line

Every stage from arbitration (6) through compounding is **wired by code evidence and has never been
runtime-exercised in this audit's history** — not because of an execution bug, but because Stage 5
(PERSIST_GATE + a 0.9500 ConfFloor) has, in every session observed, blocked 100% of candidates before
they could reach Stage 6. That gate is itself a correct, evidence-based response to Stage 3's proven
lack of predictive edge. The practical consequence: this system's trade lifecycle is currently a
**strategy-throughput problem gated correctly at Stage 5, not a plumbing-correctness problem** —
tightening or loosening any gate downstream of Stage 5 does not change the fundamental fact that Stage 3
has no proven edge to gate. See `daily_target_assessment.md` and `strategy_contribution.md` for the
implications of this for any daily-trade-count or profit target.
