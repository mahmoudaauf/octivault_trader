# Binance earning programs — complete audit

*2026-09-13. Every program probed against the live account via API. This exists
so no future session re-asks "is there something better on Binance"; the answer
is measured, not assumed.*

## Verdict

**Simple Earn Flexible USD1 at 8.59% is the best yield available on Binance for
dollar-stable capital. Nothing else on the platform beats it.** The machine is
already in it, and the tier-filler will spread across USD1/USDT/USDC caps
automatically up to $2,600.

## What was probed, and what it pays

| program | API | verdict for a stable core |
|---|---|---|
| **Simple Earn Flexible** | ✅ | **8.59% — WHERE WE ARE. Best on platform.** |
| Simple Earn Locked | ✅ | **no stablecoin products offered to this account** |
| Staking (100 products) | ✅ | **ZERO stablecoin products.** All volatile: SENT 29.9%/90d, NEWT 29.9%, LPT 18.9%, ATOM 13%… each carries token price risk + 90-day lockup, and most read `isSoldOut: true`. The nearest stable-ish is USDS at **3.20%** — worse than 8.59%. |
| **Dual Investment** | ✅ | **101.96% APR — and EXCLUDED.** See below. |
| Launchpool | ✅ | Armed and verified; captures FDUSD at ~21.5% when a pool runs. None live, 70 completed all-time. |
| Auto-Invest | ✅ | A DCA *scheduler*, not a yield product. Nothing to earn. |
| ETH staking | ✅ | Requires ETH. Converting stable→ETH takes price risk for a lower rate. |
| Mining pool | ✅ | Requires hardware. |
| Convert | ✅ | Already used by rotation (zero-fee USD1/USDC/USDT legs). |
| Spot rebate | ✅ | `totalRecords: 0` — no fee rebates ever earned; requires referral volume. |
| Crypto loans | 🔒 | Endpoint deprecated. |
| BSwap / liquidity farming | 🔒 | HTTP 404 — retired by Binance. |
| VIP loan | ✅ | Institutional; not applicable. |

## Dual Investment — the one thing paying more, and why it is out

101.96% APR on BTC→USDT at strike $77,500 is the highest rate reachable on this
account by a wide margin. It is excluded, for a reason that is structural rather
than cautious:

**Dual Investment is selling an option.** You are the writer. Deposit BTC with a
strike above spot and you have sold a covered call: if BTC finishes above the
strike your BTC is sold at the strike and you keep the premium; if it finishes
below you keep the BTC and the premium. Deposit USDT with a strike below spot
and you have sold a cash-secured put.

The operator ruled option trading impermissible. Dual Investment is an option
contract under a different name, so it is out on the operator's own constraint,
not on a risk judgement. Recording it here so a future session does not
"discover" a 101% rate and treat the exclusion as an oversight.

The same reasoning already excluded **BFUSD** and **USDe** — both derive their
yield from a basis trade (long spot, short perp), i.e. shorting under the hood.

## What this closes

Combined with the earlier venue work, every route for dollar-stable capital is
now measured:

- **On Binance:** 8.59% is the ceiling. Audited above.
- **Off Binance, for capital above the $2,600 tier caps:** Ondo USDY at 4.65%
  (see `one_dollar_per_hour_plan.md`); Kraken and Coinbase are unavailable in
  Egypt.
- **Every trading edge:** falsified, including order-book imbalance on 57.5 days
  of data (see the addendum in the plan).

There is no unexamined yield lever left on this platform. The remaining levers
are capital and referral, both of which need operator action.
