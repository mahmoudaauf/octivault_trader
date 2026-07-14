# Profit Realization & Compounding Integrity Assessment

Date: 2026-07-14
Scope: `core_engine/native/tp_sl_engine.py`, `core_engine/native/executor.py`,
`core_engine/native/order_execution.py`, `core_engine/native/capital_allocator.py`,
`core_engine/native/daily_compounding.py`, `core_engine/native/nav_protection.py`,
`core_engine/native/fill_tracker.py`, `core_engine/native/polling_coordinator.py`,
`core_engine/native/position_hydration_engine.py`, `core_engine/native/config_loader.py`.

This is additive to the existing `docs/audit/` set. It does not repeat the prior
inventory; it verifies a specific claim implicit in that inventory ("TP/SL, capital
allocation, daily compounding are wired") down to whether the numbers those
components act on are actually **net-of-cost** and **non-double-counted**. One
correction to the prior audit is flagged in §6.

---

## 1. Bottom line

**A human cannot currently trust a "net profitable trade" number from this system's
own logs or state.** The only per-trade P&L the live (default-config) path ever
computes is **gross** and it is never persisted anywhere queryable — it exists for
one `logger.info` line and then is gone. The one component that *does* compute a
correctly fee-adjusted realized P&L and write it into `shared_state.metrics` — the
Tier-2 `NativeFillTracker` — is **idle by default** (`polling_enabled=True` disables
it in favor of `NativePollingCoordinator`, which does not replicate the calculation).
Any claim of "net profitable" would require independent reconciliation against
Binance's own trade/fee history (`get_my_trades`, summed `commission` per
`commissionAsset`, converted to USDT) — the system does not do this reconciliation
for you today.

---

## 2. The net-realized-profit formula: does it exist as a single computation?

**No single, authoritative implementation exists.** Fee/slippage/cost accounting is
scattered across at least four different formulas that disagree on what "profit" a
trade produced, and none of them is: `gross proceeds − entry fees − exit fees −
slippage − funding/other costs`.

| Location | Formula | Fee-aware? | Slippage-aware? | Reaches shared state? |
|---|---|---|---|---|
| `executor.py::_close_position` (`TRADE_CLOSED` log, line ~674) | `_gross_pnl = (fill_price - entry_price) * qty` | No | No | **No** — logged only, honestly labeled `gross_pnl`, never written to `shared_state.metrics` or the trade journal payload |
| `fill_tracker.py::_handle_sell_fill` (line ~318) | `realized_pnl = (fill.price - entry_price) * qty - commission_quote`, where `commission_quote` correctly converts `fee_asset` (e.g. BNB) to quote via `price_cache` (`_commission_quote`, line 183) | **Yes** (commission only) | No | Yes — `shared_state.metrics["realized_pnl"] += realized_pnl` — **but this code path is idle in the default runtime config** (see §3) |
| `position_hydration_engine.py::_calculate_pnl` (line ~594) | `unrealized_pnl = current_value - entry_value - fees_paid`, where `fees_paid = sum(fill.get("fee") or fill.get("commission"))` **without checking `commissionAsset`** | Partially — subtracted from *unrealized*, not *realized*; and the fee value is taken at face value regardless of the asset it was actually denominated in | No | Yes, at startup hydration only |
| `tp_sl_engine.py` time-force-exit gate (line ~476) | `_net_profit_pct = (current_price - entry_price)/entry_price - round_trip_cost_pct`, where `round_trip_cost_pct = max(0.002, (taker_fee_bps*2 + exit_slippage_bps)/10000)` | Yes, but as a **configured estimate**, not the actual fill's fee/slippage | Yes, same caveat | Used only to decide whether to force-exit; not persisted as a trade's P&L |

Consequences of the fragmentation:

- The number that actually reaches `shared_state.metrics["realized_pnl"]` — the one
  field `NAVAttributionEngine`, `ObjectiveFeedbackController`, and
  `AdaptiveCapitalEngine` all read (`nav_protection.py:73`,
  `objective_feedback_controller.py:368`, `adaptive_capital_engine.py:99`) — is
  produced exclusively by `fill_tracker.py`. Under the default production
  configuration that code path never runs (§3), so `metrics["realized_pnl"]`
  **stays at its initialized value of `0.0` for the life of the process**
  (`shared_state.py:103`).
- The only place a human sees a per-trade net-of-fee number is the `tp_sl_engine`
  force-exit decision, and that number never appears in a log line or a journal
  entry — it is purely internal to a boolean gate.
- `position_hydration_engine.py` mixes fee currency: Binance returns `commission`
  denominated in `commissionAsset` (often BNB when the fee-discount is enabled, or
  the quote asset otherwise). `fill_tracker.py` explicitly converts
  (`_commission_quote`, checking `fee_asset == quote_asset` vs `base_asset` vs a
  cross-price lookup); `position_hydration_engine.py` does **not** — it sums the raw
  numeric `commission` regardless of what asset it was paid in. If any historical
  fill paid its fee in BNB, the `fees_paid` figure produced at every restart is a
  unit-mismatched number (BNB units added into a USDT-denominated field), silently
  corrupting the post-restart `unrealized_pnl` for every hydrated position that had
  a BNB-denominated fee in its trade history.
- `order_execution.py::_extract_avg_fill_price` reads Binance's `fills[]` array
  (which carries `commission`/`commissionAsset` per fill) only to compute the
  volume-weighted average *price* — it discards the commission fields entirely.
  Fee data available directly on the order-placement response is never surfaced to
  the executor at all; the executor has zero fee visibility on either entry or exit.

**Conclusion for this sub-question:** no, the net-realized-profit formula "gross
proceeds − entry fees − exit fees − slippage − other costs" is not implemented as a
single computation anywhere in the codebase. Fee handling is scattered across four
places with three different definitions of "cost," only one of which is fee-aware in
a currency-correct way, and that one is not reachable in the default runtime.

---

## 3. Why `fill_tracker.py`'s correct calculation doesn't run

Per the existing audit (`component_inventory.md` row 18, verified unchanged):
`NativeFillTracker` is constructed only when `polling_enabled=False`
(`bootstrap.py:529` legacy branch or `bootstrap.py:638` fallback). The default value
of `polling_enabled` is `True` (`bootstrap.py:117,227`), so under default
configuration `fill_tracker` stays `None`, its `.start()` is never called
(`orchestrator.py:161-162` guards on `fill_tracker is not None`), and
`NativePollingCoordinator` substitutes for fill detection instead.

Confirmed this session: `polling_coordinator.py` has **no** `realized_pnl`,
`commission`, or fee-bps computation anywhere in the file (only two `metrics[...]`
writes exist, both unrelated timestamp/session-elapsed fields — verified by direct
grep). So the substitute component performs fill detection and position bookkeeping
but does not replicate the P&L/fee calculation that `fill_tracker.py` would have
done. This is a real functional gap, not merely a redundant-but-idle module as the
"intentional design" framing in `component_inventory.md` implies for fill *tracking*
— the fee-aware P&L side-effect that rides along with `fill_tracker`'s fill handling
has no equivalent in the polling substitute.

---

## 4. TP/SL engine (`tp_sl_engine.py`)

- **TP/SL calculation**: ATR-based, volatility-adaptive (`calculate_tp_sl`, line 315).
  Sound design — SL floors/ceilings (1.0%–2.5%), TP enforces min 2:1 R:R and a hard
  1.5%–6% band. This *is* fee-aware at the margin level (`_MIN_NET_PROFIT_PCT`,
  `_ROUND_TRIP_FEE_PCT` class constants, and the constructor-computed
  `_round_trip_cost_pct` from `taker_fee_bps` / `exit_slippage_bps`), but it is a
  configured estimate of round-trip cost, not a measurement of the actual fill's
  fee/slippage. If real fees/slippage exceed the configured estimate (e.g. taker
  instead of maker fill, wider slippage in a fast market), the TP floor is silently
  insufficient and a "TP_HIT" exit can still be net-negative after real costs.
- **Trailing stop**: regime-aware activation/distance (`_REGIME_TRAIL_PARAMS`),
  reasonable design, correctly distinguishes trend vs. chop.
- **Time-based force exit**: `_AGE_FORCE_EXIT_SEC` (3h default) with a profitability
  carve-out (`check_triggers` line 468) that uses the same estimated
  `_round_trip_cost_pct`, not actual costs.
- **Protective tightening** (`_compute_protective_tightening`, line 826): this is the
  one place TP/SL logic explicitly reacts to `nav_protection_state` (via
  `allow_tp_sl_adjustment`, `suggested_actions`, `protection_mode`). **This branch is
  permanently a no-op in production** — see §5, `nav_protection_state` is always
  `{}` because `nav_protection.py`'s engines are never invoked. `allow_tp_sl_adjustment`
  therefore defaults to `False` (`bool(nav_protection.get("allow_tp_sl_adjustment", False))`,
  line 611) and `_compute_protective_tightening` returns `None` unconditionally
  (guard at line 841). The dynamic-widen path (`_maybe_widen_tp`) still functions
  independently of NAV protection.
- No slippage/fee is deducted from the TP/SL trigger price itself when checking
  `current_price >= tp` / `current_price <= sl` — triggers fire on mark price, and
  the actual fill can differ from mark price by the executor's observed
  `avg_slippage_bps` (which the executor itself tracks in `_record_execution_quality`
  but which `tp_sl_engine.py` never reads back).

---

## 5. `nav_protection.py` — CORRECTED: actually wired via `main.py`, not unwired

**Correction, 2026-07-14 (remediation pass).** This section originally claimed
`nav_protection.py` was "confirmed unwired, not merely candidate unwired," based on a
grep scoped to `core_engine/native/` only (reproduced below, unedited, as a record of
the mistake):

```
$ grep -rn "evaluate_nav_protection\|update_nav_protection\|NAVProtectionEngine\|NAVAttributionEngine\|nav_protection_state" core_engine/native/ | grep -v /test
core_engine/native/arbitration_engine.py:369:  _nav_prot = getattr(self._shared_state, "nav_protection_state", {}) or {}   # READ only
core_engine/native/nav_protection.py: (definitions only)
core_engine/native/shared_state.py:123-142: (field + setter definitions only)
core_engine/native/tp_sl_engine.py:606:  nav_protection = getattr(self.shared_state, "nav_protection_state", {}) or {}  # READ only
```

**That grep's directory scope was the bug.** `main.py` — which is outside
`core_engine/native/` and was never included in the search — calls
`evaluate_nav_protection(_ss)` directly at `main.py:412-433`, gated on a 60-second
interval (`_nav_prot_due`), using the exact same `NativeSharedState` instance
(`_ss = getattr(_native_orch, "_shared_state", None)`, `main.py:282`) that
`arbitration_engine.py:369` and `tp_sl_engine.py:606` read from. This is the same
class of mistake as the original arbitration-engine false positive earlier in this
audit series: a static trace that checked the "obvious" files (here,
`core_engine/native/`) and missed the actual call site living in `main.py`.

**Verified directly this pass** (not just re-read the code): calling
`evaluate_nav_protection()` against a realistic `NativeSharedState` instance completes
with no exception and populates `nav_protection_state` correctly. The absence of any
`[main:NAV-PROTECT]` log line in the observed live session's `logs/run_latest.log` is
fully explained by `main.py:419`'s own logging being conditional (`if _mode != "NORMAL"`
or a new peak) — NAV was flat at 57.85 the entire session, so `mode` stayed `NORMAL`
and nothing logged. This is silent-by-design correct behavior, not evidence of failure.

**Corrected conclusion:** `nav_protection.py` IS wired into the live decision path —
`allow_tp_sl_adjustment`, `suggested_actions`, and drawdown-tiered protection modes
(`DEFENSIVE`/`FREEZE_BUY`/`RECOVERY`) are live and reactive to real NAV state, not dead
code. Remediation item #7 (in `remediation_plan.md`, "wire `evaluate_nav_protection()`
into the orchestrator's periodic cycle") is **withdrawn** — no code change was made
based on it. This correction does NOT change §6's separate point about
`daily_compounding.py` lacking a `protected_profit_reserve` term — `nav_protection.py`
being wired doesn't by itself mean its `locked_profit_usdt`/`protection_floor_usdt`
outputs are yet *connected* to the compounding formula (see remediation item #17,
which still applies, now as "connect two already-wired systems" rather than "wire an
unwired one first").

---

## 6. Compounding path (`capital_allocator.py` + `daily_compounding.py`)

**What it actually reads:** `NativeCapitalAllocator.allocate_for_buy()` calls
`self._pm.get_nav()` → `NativePortfolioManager.get_nav()` → returns
`shared_state.nav_usdt` if set, else `free_balance + Σ(position value at mark
price)` (`portfolio_manager.py:76-82,84`). **This is mark-to-market total account
value, i.e. it includes unrealized P&L on every open position.** It is not a
reconciled realized-profit ledger.

**What `DailyCompoundingPolicy.sizing_nav()` does with that number**
(`daily_compounding.py:38`): it is a genuinely good mitigation for the "size off
unrealized gains" risk, structured as follows:
- It freezes a `sizing_nav_usdt` snapshot for the UTC day.
- A new day's NAV is only "committed" (allowed to raise the sizing base) if the
  portfolio is flat (`has_open_positions=False`) at the moment of rollover check —
  if flat, `nav_usdt` at that instant *is* effectively realized cash, so the
  compounding base only ratchets up off genuinely settled capital.
- If the portfolio is not flat at rollover, the old `sizing_nav_usdt` stays frozen
  (`pending_rollover=True`) and is retried on the next check.
- Losses are NOT held back: `return min(current_nav, self.state.sizing_nav_usdt)` —
  any intraday/unrealized loss immediately caps the *sizing* NAV downward, even
  though the committed `sizing_nav_usdt` floor itself doesn't move down until a
  flat rollover. This means a losing streak reduces order sizes immediately without
  waiting for a day boundary, which is the conservative direction to be wrong in.

**Where it falls short of `compoundable_profit = max(0, reconciled_realized_net_profit
− protected_profit_reserve)`:**
- There is no `protected_profit_reserve` concept feeding into `daily_compounding.py`
  at all — it interacts with nothing from `nav_protection.py` (which, per §5, would
  have been the natural source of a reserve/floor concept, and is unwired anyway).
- The "reconciled realized net profit" term doesn't exist here — the mechanism
  never asks "how much of this NAV increase came from realized, fee-netted trade
  P&L" vs. "how much is currently-unrealized mark-to-market on still-open
  positions." It sidesteps that question entirely by only committing NAV when
  the portfolio is flat (no open positions → no unrealized component to
  misattribute). That's a reasonable structural workaround for **is compounding
  ever inflated by unrealized PnL** (answer: no, by construction, given flat-only
  commit), but it is not the same guarantee as reconciling against actual
  fee/slippage-netted realized P&L — a day that closes flat after a string of
  fee-heavy churn trades will show whatever `free_balance_usdt` the exchange
  reports, which is correct (exchange balances already reflect fees deducted), but
  the system itself never independently verifies that number against its own
  trade-level fee bookkeeping (which, per §2, doesn't reliably exist under default
  config anyway).
- **Double-counting is not structurally possible** in the compounding calculation
  itself — `sizing_nav()` takes one exchange-derived NAV number and clamps it; there
  is no path where the same realized profit is added into the sizing base twice.
  The risk here is not double-counting, it's **unverifiable premise**: the
  mechanism is only as trustworthy as `shared_state.nav_usdt`/`free_balance_usdt`,
  which are themselves populated by the balance poller reading the exchange
  directly (not derived from the broken internal `metrics["realized_pnl"]`), so this
  part is actually more trustworthy than the internal P&L bookkeeping in §2 — it
  just isn't "reconciled realized net profit" in the sense the audit brief asks
  about, it's "current exchange-reported account value, gated to not compound
  intraday/unrealized swings."

**Verdict on `compoundable_profit = max(0, reconciled_realized_net_profit −
protected_profit_reserve)`:** not implemented, not even partially in name, but
`daily_compounding.py`'s flat-rollover NAV-freeze achieves a similar practical
effect for the specific failure mode "compounding on unrealized gains" — at the
cost of not implementing (or being able to report) a genuine realized-P&L figure,
and with zero connection to a profit-reserve/floor concept.

---

## 7. Config-name divergence: verified impact on PnL trust

Per the existing audit's flagged concern (`current_state_assessment.md`), this pass
traced the actual wiring:

| Concept | BootstrapConfig (env var → field) | config_loader.py (env var → key) | Which one is live? |
|---|---|---|---|
| Round-trip fee | `TAKER_FEE_BPS`/`MAKER_FEE_BPS` → `taker_fee_bps`/`maker_fee_bps` (default 10.0 bps each) | `EXIT_FEE_BPS` → `exit_fee_bps` (default 10.0) | **BootstrapConfig** — feeds `tp_sl_engine.py:86-97`'s `_round_trip_cost_pct`. `config_loader.py`'s `exit_fee_bps` is unread by any production code. |
| Exit slippage | `EXIT_SLIPPAGE_BPS`/`CR_PRICE_SLIPPAGE_BPS` → `exit_slippage_bps` | *(no equivalent key)* | BootstrapConfig only |
| Take-profit | `TP_PCT` → `tp_pct` (default 0.03) | `TAKE_PROFIT_PCT` → `take_profit_pct` (default 2.0) | Neither field is actually consumed by `NativeTPSLEngine` — TP/SL is computed dynamically from `TP_ATR_MULT`/`SL_ATR_MULT`, not from a static `tp_pct`/`sl_pct`. `bootstrap.py`'s `tp_pct`/`sl_pct` fields exist on `BootstrapConfig` but appear unused by the live TP/SL path (confirmed: `tp_sl_engine.py` never reads `config.tp_pct` or `config.sl_pct`). |
| Stop-loss | `SL_PCT` → `sl_pct` (default 0.02) | `STOP_LOSS_PCT` → `stop_loss_pct` (default 5.0) | Same as above — neither reaches the ATR-based SL calc. |
| Compounding | `DAILY_COMPOUNDING_ENABLED` → `daily_compounding_enabled` (default `True`), threaded through `bootstrap.py:245,828` into `NativeCapitalAllocator` | `COMPOUNDING_ENABLED` → `compounding_enabled` (default `true`) | **BootstrapConfig only** — `capital_allocator.py` constructs `DailyCompoundingPolicy(enabled=daily_compounding_enabled, ...)` directly from the `BootstrapConfig` value passed at `bootstrap.py:828`. |

**Key finding, correcting the severity (not the existence) of the prior audit's
concern:** `core_engine/native/config_loader.py`'s `ConfigLoader`/`get_config()` is
**not called from any live code path**. It is exported from
`core_engine/native/__init__.py` but the only other references to it in the
repository are inside `_archive/2026-06-19_legacy_system/` (archived, not run).
Direct greps for its output keys (`STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`, `EXIT_FEE_BPS`)
confirm zero non-archived, non-test consumers. **So the two config systems do not
currently diverge in effect** — only `BootstrapConfig` is live, `config_loader.py` is
dead weight. The operational risk this still leaves is narrower than "silent gating
divergence": it's an **operator trap** — someone tuning fees/TP/SL/compounding by
setting `EXIT_FEE_BPS`, `TAKE_PROFIT_PCT`, `STOP_LOSS_PCT`, or `COMPOUNDING_ENABLED`
(all plausible-looking env var names) would see **zero effect** on the running
system and might reasonably but wrongly conclude their change was applied — directly
undermining trust in any subsequent PnL number if they believed they'd, say,
tightened the fee assumption or disabled compounding. This should be corrected in
`component_inventory.md`/`current_state_assessment.md`: the risk is real but its
mechanism is "dead code masquerading as live config," not "two live systems
disagreeing."

---

## 8. Can a human currently trust a "net profitable trade" claim from this system?

**No, not without independent reconciliation.** Concretely, to verify any trade or
session was net-profitable today, a human would need to bypass every in-process
number and instead:

1. Pull `GET /api/v3/myTrades` per symbol directly (as `position_hydration_engine.py`
   already does, but correctly handling `commissionAsset` — which that module does
   not).
2. Sum `commission` per trade, converting non-quote-asset fees (e.g. BNB) to USDT
   at the trade-time price.
3. Compute `Σ(sell_proceeds) − Σ(buy_cost) − Σ(fees_in_quote)` per closed round trip.
4. Separately estimate slippage vs. some reference price series, since the system
   tracks `avg_slippage_bps` as a rolling *average* execution-quality metric
   (`executor.py::_record_execution_quality`) but never attributes it to individual
   trades' P&L.

None of steps 1-4 happen automatically today. The closest in-process artifact is the
`TRADE_CLOSED` log line in `executor.py`, and it is explicitly gross
(`gross_pnl=%.4f USDT`) — which is at least honestly labeled, so it should not be
mistaken for net P&L, but it also isn't captured anywhere queryable (it's a log
line, not a metric, not a journal field, not a database row).

---

## 9. Summary of concrete defects found this pass

1. **`metrics["realized_pnl"]` is dead in the default runtime.** Its only writer
   (`fill_tracker.py`) is disabled by default (`polling_enabled=True`); its
   substitute (`polling_coordinator.py`) does not replicate the calculation. Every
   downstream consumer of this metric (`NAVAttributionEngine`,
   `ObjectiveFeedbackController`, `AdaptiveCapitalEngine`) is operating on a
   permanently-zero realized-PnL signal.
2. **CORRECTED (2026-07-14 remediation pass): `nav_protection.py` is wired, not
   unwired.** The original claim here was based on a grep scoped only to
   `core_engine/native/`, which missed `main.py:412-433`'s direct call to
   `evaluate_nav_protection()` on a 60s cadence against the live shared-state
   instance — see §5 for the full correction. Verified directly: calling it against
   a realistic shared state populates `nav_protection_state` and produces
   `allow_tp_sl_adjustment=True`/`suggested_actions=['REDUCE_BUY_SIZE',
   'TIGHTEN_TP_SL']` under a drawdown scenario. This is the module that contains the
   codebase's only real "protected profit reserve" concept (`locked_profit_usdt`,
   `protection_floor_usdt`), and it does have real runtime effect — it just isn't yet
   *connected* to the compounding formula (that gap is real, see item 7 below).
3. **`tp_sl_engine.py`'s protective-tightening branch is live, not a no-op** — direct
   consequence of the (2) correction. `allow_tp_sl_adjustment` does go `True` under
   drawdown, confirmed by direct invocation this pass.
4. **`position_hydration_engine.py` mis-sums fees** across `commissionAsset`
   boundaries (unlike `fill_tracker.py`, which does this correctly) — a real
   currency-unit bug in the one place expected to reconstruct authoritative
   post-restart P&L.
5. **No component computes or persists true net realized P&L per trade** — the only
   per-trade number available (`executor.py`'s `TRADE_CLOSED` log) is gross, and it
   is not written to `shared_state`, the trade journal payload, or any metric.
6. **Config divergence is real but currently inert**: `config_loader.py`'s fee/TP/SL/
   compounding env vars are dead code (unreferenced outside `_archive/` and tests) —
   downgrade the "live risk" framing in the existing audit to "operator trap /
   maintenance hazard," not "two systems actively disagreeing."
7. **Daily compounding (`daily_compounding.py`) is structurally sound** for its
   narrow purpose (never compounds an unrealized/intraday gain) via the flat-day
   rollover gate, and cannot double-count, but it does not implement — and has no
   connection to — a `protected_profit_reserve` or reconciled-realized-profit
   concept; it substitutes "exchange-reported NAV at a flat-portfolio instant" for
   that concept, which is a reasonable but different guarantee.

---

## 10. Recommended remediation (feeds `remediation_plan.md`)

- Either enable `fill_tracker.py` in the default polling-mode configuration, or port
  its `_commission_quote`-based realized-PnL calculation into
  `polling_coordinator.py`'s fill-handling path, and have it write
  `shared_state.metrics["realized_pnl"]` on every SELL fill.
- Fix `position_hydration_engine.py`'s fee summation to check `commissionAsset` the
  same way `fill_tracker.py` does (or share one helper between them).
- ~~Wire `evaluate_nav_protection()` into the orchestrator's periodic cycle~~ —
  **withdrawn 2026-07-14; already wired via `main.py:412-433`, see §5 correction.**
- Add a single canonical `compute_net_trade_pnl(entry_fill, exit_fill)` function
  that subtracts entry fee + exit fee + observed slippage from gross proceeds, call
  it from `executor.py::_close_position`, and persist the result to
  `shared_state.metrics["realized_pnl"]` and the trade journal — replacing the
  current log-only `gross_pnl` line. **Done 2026-07-14**, see `changes_made.md`.
- Delete or clearly deprecate `core_engine/native/config_loader.py`'s fee/TP/SL/
  compounding keys (or wire them as aliases into `BootstrapConfig` with a startup
  warning if both are set and disagree) to remove the operator trap.
- Once realized P&L is trustworthy, connect `nav_protection.py`'s
  `locked_profit_usdt`/`protection_floor_usdt` to `daily_compounding.py` as the
  actual `protected_profit_reserve` term in a `compoundable_profit = max(0,
  reconciled_realized_net_profit − protected_profit_reserve)` formula, rather than
  relying solely on the flat-rollover NAV freeze.
