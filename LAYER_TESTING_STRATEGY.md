# 🧪 LAYER TESTING STRATEGY — Octivault Trader
**Companion to:** `LOGICAL_LAYERED_ARCHITECTURE.md`
**Test root:** `tests/layers/`
**Run all layer tests:** `python3 -m pytest tests/layers -v`
**Run one layer:** `python3 -m pytest tests/layers/test_l4_execution.py -v`

---

## 1. Testing principles

Every layer is tested at four progressively stricter levels:

| Level | What it verifies | Example |
|------|-------------------|---------|
| **A. Contract** | The contract object itself: required output fields, validators, enums | `L4ExecutionContract().validate_output(...)` |
| **B. Unit (sandboxed)** | One concrete component behaves correctly with **all neighbours mocked** | `MakerExecution.submit()` against a mocked `IExchangeClient` |
| **C. Boundary** | The interface between layer N and N±1 — only the typed `LayerInput` / `LayerOutput` envelope is allowed | feed a fake L3 `IPortfolioAuthority` into L4, verify L4 only calls allowed methods |
| **D. Invariant** | Hard rules from the architecture doc — must *always* hold | `CASH + TRADING + EXTERNAL == WALLET_TOTAL`; no order without a `ReservationToken` |

Every layer has a dedicated file under `tests/layers/` that runs all four levels for that layer.

---

## 2. Expected outcome per layer

| L# | Layer | The single sentence its tests must prove |
|----|-------|-------------------------------------------|
| L0 | Cross‑Cutting | *Pure: same input ⇒ same output, zero I/O, no state mutation.* |
| L1 | Exchange I/O | *Every byte to/from the exchange goes through here, retried, and surfaces as a typed L0 object or a typed `ExchangeError`.* |
| L2 | Wallet & Market Data | *A `WalletSnapshot` is exchange‑verified, immutable, and `CASH+TRADING+EXTERNAL == WALLET_TOTAL`.* |
| L3 | Portfolio & State | *Every position change is journaled before becoming visible, and capital is only spendable through `ReservationToken`.* |
| L4 | Execution | *No order is sent without a valid `ReservationToken`; every fill produces exactly one journal entry.* |
| L5 | Strategy | *Pure decisions: same `PortfolioCtx` ⇒ same `TradeIntent` set; never touches exchange or capital directly.* |
| L6 | Governance | *Every vetoed intent records a typed reason; no silent downsizing; caps are inviolable.* |
| L7 | Observability | *Read‑only: a forced exception in any L7 component never propagates to L1–L6.* |
| L8 | Lifecycle | *Boot order is exactly `L0→L1→L2→L3→L4→L6→L5→L7`; a single layer can be restarted in place; required‑layer failure aborts boot.* |

---

## 3. Standard test fixtures

Each layer test file imports four standard fakes from `tests/layers/fakes.py`:

- `FakeExchange` — implements `IExchangeClient`. Programmable balances, fills, latencies, errors.
- `FakePortfolio` — implements `IPortfolioAuthority`. Tracks reservations.
- `FakePolicyGate` — implements `IPolicyGate`. Either approves all, vetoes all, or by predicate.
- `FakeMetrics` — implements `IMetricsSink` + `IAlertBus`. Records calls for assertion.

Tests **never** import real network clients, real WS feeds, real DB. The contracts make this enforceable.

---

## 4. Layer-by-layer test plan

### L0 — Cross-Cutting (`test_l0_cross_cutting.py`)
- Determinism: `pnl_calculator(x) == pnl_calculator(x)` across runs.
- No I/O: monkey-patch `socket`, `open`, `requests.*` to raise; L0 must still import and run.
- Contract objects (`OctiError` hierarchy, `Position`, `Signal`) are constructible.

### L1 — Exchange I/O (`test_l1_exchange.py`)
- Contract: `L1ExchangeContract.validate_output` rejects missing fields.
- Retry: a flaky `FakeExchange` (fails twice, then succeeds) yields one successful call.
- Order cache: `OrderCacheManager.upsert` then `reconcile` produces 0 drift.
- Boundary: L1 raises typed `ExchangeError` on permanent failure (never bubbles raw `requests.HTTPError`).

### L2 — Wallet & Market Data (`test_l2_wallet.py`)
- Contract: `L2WalletContract.validate_output` requires `assets`, `positions`, `last_updated`.
- Snapshot immutability: mutating returned dict does not affect next snapshot.
- Classification: `EXTERNAL_POSITION` survives a sync (read‑only invariant).
- OHLCV cache: cache hit on second identical query.

### L3 — Portfolio & State (`test_l3_portfolio.py`)
- Bucket conservation: `CASH + TRADING + EXTERNAL == WALLET_TOTAL` after any update.
- Reservation: `reserve(qty)` then `release(token)` returns capital exactly to `CASH`.
- Journal: every `apply_fill()` produces exactly one journal entry, ordered by ts.
- Read-only EXTERNAL: any attempt to mutate an EXTERNAL position raises `PortfolioInvariantError`.

### L4 — Execution (`test_l4_execution.py`)
- Contract: `L4ExecutionContract.validate_intent` rejects missing `reservation_token`.
- No-reserve, no-order: `submit()` without a token never calls `FakeExchange.place_order`.
- Exactly-once journaling: a duplicate `submit()` for the same intent produces 1 journal row.
- Failure path: exchange returns ERROR ⇒ reservation is released, no fill emitted.

### L5 — Strategy (`test_l5_strategy.py`)
- Purity: `propose(ctx)` is deterministic with a fixed RNG seed.
- No side effects: `propose()` never calls `place_order` (assert via mock).
- Fusion: with two opposing agent signals, `signal_fusion` collapses to the higher‑confidence side.
- Feedback: `feedback(fill)` updates internal stats but never returns a value.

### L6 — Governance (`test_l6_governance.py`)
- Veto: any intent breaching `max_position_pct` returns a `GovernanceVeto` with a reason.
- No silent downsize: if veto threshold is hit, the result is veto, **not** a smaller order.
- Override journaling: `override(rule_id, value)` produces a journal entry with old/new value.
- Cap inviolability: random fuzz of 1000 intents never produces an approved order above caps.

### L7 — Observability (`test_l7_observability.py`)
- Read-only: any attempt by an L7 component to call a mutating method on lower fakes raises in test mode.
- Failure containment: forcing `MetricsRegistry.gauge` to raise still allows L4→L3 to commit a fill.
- Subscription: the alert bus delivers events in the order they were emitted.

### L8 — Lifecycle & Recovery (`test_l8_lifecycle.py`)
- Boot order is exactly the spec list (already covered in `test_layered_architecture.py`).
- Single-layer restart: stop+start of L4 leaves L1‑L3 untouched.
- Required-layer failure aborts boot.
- Watchdog: when the health probe of a layer flips to `DOWN`, the watchdog calls `restart_layer` once.

---

## 5. CI matrix

```yaml
# pseudo-CI config
- name: lint-layers
  run: python3 scripts/check_layer_imports.py            # 0 new violations
- name: layer-contracts
  run: python3 -m pytest tests/test_layered_architecture.py
- name: layer-tests
  run: python3 -m pytest tests/layers -q
- name: full-suite
  run: python3 -m pytest tests -q
```

Any commit that lowers the per‑layer pass count or introduces a new layer-import violation is rejected at PR time.
