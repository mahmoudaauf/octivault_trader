# Hourly profit evidence and monitoring

Implemented September 8, 2026. The target is **$1 net per calendar hour from
$60.65**, which requires 1.6488% per hour. The system has not demonstrated that
return. These changes measure the gap and collect new evidence; they do not
enable a strategy or increase the real allocation.

## What is running

- `profit_research_cycle.py` runs every 15 minutes through the local launchd
  job `com.octivault.profit-research` while the Mac is available.
- `capital_paper.py` starts a separate forward experiment at exactly $60.65
  of **virtual cash**, with at most 20% in a single spot position. It uses
  public market data only, does not load credentials, and contains no live
  order or transfer methods.
- `profit_readiness.py` refreshes the report each cycle. Exchange fill history
  is refreshed hourly using authenticated **read-only** calls. A failed refresh
  is reported; cached data is not relabeled as new.
- The existing live allocator remains configured with `HYBRID_OBJECTIVE=allocate`.
  It was not restarted. The additional execution-evidence fields in its source
  take effect on its next normal process start; existing stored trades remain
  unchanged.

Latest readable report: `logs/profit_readiness/latest.md`.
Machine-readable evidence: `logs/profit_readiness/latest.json`.
Private exchange cache: `logs/profit_readiness/exchange_fills.json`.
Virtual cash, inventory, closed trades and marks: `logs/profit_readiness/forward_paper.json`.
These runtime files stay under the repository's ignored `logs/` directory.

## Findings from the historical account

The legacy ledger contains 17 live closes and six winners. Its configured-fee
estimate totals $0.032452. Unique time/price matches to exchange orders, actual
commission quantities, and historical BNB minute-price valuation bounds give
approximately **$0.095329–$0.095510 realized net** for the matched strategy
trades. Attribution remains inferred because the legacy records lack order IDs.
It is not a tax cost-basis statement or a whole-account return.

The audit also identifies an INJ exit of 1.58 units where the old ledger booked
1.59; the remaining 0.01 units retain allocated cost and are not counted as sold.
The BNB round trip used an existing BNB buffer to cover base-denominated fees;
the strategy calculation charges those fees at the BNBUSDT execution prices.
The separate account FIFO calculation is a diagnostic only: historical Earn,
transfers and cross-symbol fee-token consumption are not fully reconstructed.

At the conservative end of the strategy valuation, profit factor is about 1.073.
An additional **10 basis points (0.10%) of round-trip execution cost** changes
the historical net result to roughly **-$0.035**. The small positive result is
therefore not a demonstrated robust edge.

Some older NAV windows cross changes in wallet coverage. The report flags these
as noncomparable instead of calling newly visible assets profit. Other windows
subtract recorded net contributions, but still include market price changes.
Independent reconciliation of every deposit, withdrawal, Earn movement and
wallet remains outstanding; NAV growth is not labeled realized trading profit.

## Accounting changes

`hybrid_allocator.py report` now excludes transfers, rotation records and paper
trades, reports dollar P&L, and no longer sums trade percentages as an account
return. New close records label estimated fees explicitly and apply fees to
each side's notional. Entry/exit order IDs and available raw fills are retained
for future reconciliation; when exit evidence exists, actual sold quantity is
used. These are accounting changes, not a strategy change.

`exchange_trade_audit.py` paginates by trade ID, deduplicates fills, groups partial
fills by order, handles quote/base commissions, and leaves unknown cost basis
or unconverted commissions unresolved. Third-currency commissions for matched
strategy trades use the historical minute's low/high price, not today's price.
Execution slippage is already in real fill prices and is not deducted twice.
The extra cost stress is a hypothetical sensitivity test, clearly separate.

API references: [Binance account trade-history parameters](https://binance.github.io/binance-connector-js/classes/_binance_spot.SpotRestAPI.RestAPI.html#myTrades)
and [Binance account trade/commission fields](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/ws-api/account).

## Forward experiment and acceptance

The frozen baseline is the existing long-only 20-bar Donchian breakout with
2x volume confirmation, a 4% take-profit, 2% stop, 48-hour time exit and one-hour
reentry cooldown. **Its earlier research failed.** This run is a prospective
baseline, not a claim that the rule is newly profitable. There is no parameter
search or retrospective replay counted as forward profit. Changing parameters
requires a separate experiment state.

The simulation uses observed asks to enter and bids to exit, 0.10% fees per side,
0.05% additional slippage per side, shared cash, lot sizes and minimum notionals.
It marks open holdings at estimated liquidation value including exit costs.
An exit below the minimum remains inventory, not fabricated cash. A 2% daily
loss threshold blocks further entry; a 10% marked drawdown halts the experiment.
Stops execute only at observations and can overshoot; unseen intraday moves,
depth/queue effects and changing market-order average-price filters are not fully
modeled. Downtime exceeding 30 minutes is recorded as a coverage gap.

The report screens for 30 complete days, 100 closes, positive expectancy, profit
factor at least 1.2, a positive lower daily-bootstrap bound, resilience to the
extra 10-bps cost, and drawdown limits. These are research thresholds, not proof
or financial guarantees. Forward evidence must also include marked inventory,
fresh observations and no coverage gaps. Live accounting attribution is a
separate gate. **Passing a screen never automatically enables real trading.**

The $1/hour target is evaluated separately over total calendar time, including
idle hours. Neither a single winning trade nor a high APR snapshot meets it.

## Commands

From the project root:

```sh
.venv/bin/python3 profit_readiness.py
.venv/bin/python3 profit_readiness.py --refresh-exchange
.venv/bin/python3 profit_research_cycle.py
.venv/bin/python3 -m pytest tests/test_profit_readiness.py tests/test_capital_paper.py tests/test_hybrid_allocator.py tests/test_strategy_validation.py -q
```

The installed job is `~/Library/LaunchAgents/com.octivault.profit-research.plist`.
Its repository copy lives in `deployment/launchd/`. Stop only this research job:

```sh
launchctl bootout "gui/$(id -u)/com.octivault.profit-research"
```

This does not stop or change the live allocator. Report process exit code zero
means the diagnostic completed, not that the profit objective was achieved.
