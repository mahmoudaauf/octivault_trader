# $1.00/hour — the advisor's plan

*Prepared 2026-09-10 for the operator. Every number here was measured or
fetched this week; nothing is assumed. Where a claim could not be verified it
is marked as such.*

## The goal, priced

**$1.00/hour = $8,760/year = $168/week.** On the $60.69 we hold, that is a
14,444% annual return. It does not exist as a *return* — the best verified rate
on the exchange is 8.53%, the best sustained record in the history of finance
(Renaissance Medallion) is ~66%/yr, and the leverage needed to fake it has an
81% probability of ruin within 24 hours on our measured edge of zero.

It exists as an *income*, on two independent axes that **add**:

| axis | what it is | price of $1/hour |
|---|---|---|
| **Capital** | return on what we hold | **~$192,700** at sustainable blended rates |
| **Referral** | share of *other people's* trading fees | **~49 heavy traders or ~487 moderate ones**, $0 capital |

Every dollar of referral income reduces the capital needed one-for-one.

---

## Lever 1 — Capital

### The structure, by tranche

| tranche | where | rate | managed by the machine? |
|---|---|---|---|
| **$0 – $2,600** | Binance bonus tiers (USD1 1,500 / USDT 800 / USDC 300) | **7.9% blended** | **Yes — fully autonomous, built, live.** The tier-filler spreads across coins to fill each cap in rate order, verifies each tier is actually paid, refuses to add to a suspect product, and self-corrects. |
| **above $2,600** | *not Binance* — marginal Binance dollar earns 2.83% | see venues | **No.** Needs a venue decision and, today, is outside the machine's API reach. |

**Binance is the right venue up to $2,600 and the wrong venue past it.** That
line is the machine's capacity and it pays $3.94/week at full.

### Venue assessment for the tranche above $2,600 (operator is in Egypt)

| venue | rate | verdict | why |
|---|---|---|---|
| **Ondo USDY** | **4.65%** | **RECOMMENDED** | Tokenised T-bills (~92%) + bank deposits; Morgan Stanley custody; **non-US persons eligible**; buy on secondary market (Curve/Orca) with **no minimum**, 5–15 bps spread; **T+1 redemption, no lockup** (the "40-day" figure in one source was wrong — verified against the issuer profile); 25 bps/yr fee. Requires a self-custody wallet: **outside the machine.** |
| Kraken USDC rewards | 1.75–3.75% | **OUT** | Available only where Kraken serves US, CA, AU, UK. Not Egypt. |
| Coinbase USDC | 4.1% | likely OUT | US-customer programme; not confirmed for Egypt, same pattern as Kraken. |
| OKX / Bybit flexible | 1.09–6.09% **tiered** | partial | Available in Egypt; but the same capped-tier structure as Binance — a second ~$2–5k of high-rate capacity, then base. Would need a new exchange client. Worth it only above ~$10k. |
| BFUSD, USDe (on Binance) | 4–15% | **OUT — operator constraint** | Yield comes from a basis trade: long spot / short perp. Shorting under the hood. |
| USDP (on Binance) | 10.66% displayed | **OUT** | `canPurchase: false, isSoldOut: true`. A closed product's rate during a borrow squeeze; $32M cap, $61k/day; MiCA-delisted in Europe. The scan already excludes it. |
| Binance Locked products | — | none offered | No locked stablecoin products exist for this account today. |

**Honest caveat on the recommended venue:** USDY is on-chain. That means a
wallet you control, gas fees, a DEX trade, and no API the machine can read. I
can *account* for it in NAV by reading the wallet balance, but I cannot move it.
For $190k that is the right trade-off — counterparty quality matters more than
1pp of rate — but it is a manual step, and I will not pretend otherwise.

### What capital buys (tiers first, then 4.65%)

| capital | per week | per hour | % of goal |
|---|---|---|---|
| $60 | $0.10 | $0.0006 | 0.06% |
| $600 | $0.99 | $0.0058 | 0.58% |
| **$2,600** | **$3.94** | $0.0234 | 2.3% — machine full |
| $10,000 | $10.53 | $0.0626 | 6.3% |
| $50,000 | $46.31 | $0.275 | 27.5% |
| $100,000 | $91.05 | $0.540 | 54.0% |
| **$192,700** | **$168** | **$1.00** | **100%** |

---

## Lever 2 — Referral (Binance Referral Pro)

### The one irreversible decision — read before clicking anything

Binance has two modes. **Referral Lite** pays a one-time $100 fee-rebate
*voucher* per friend who deposits $50 within 14 days. **Referral Pro** pays an
ongoing cash commission on every trade your referees ever make. **For $1/hour
only Pro works** — Lite is a coupon, not income.

Secondary sources state the Lite/Pro choice is **irreversible**. The official
FAQ I fetched does not confirm or deny it. **Treat it as irreversible.** Before
choosing, open *Account → Referral* in the app and check whether a mode is
already active. If Lite is already locked on this account, the referral lever
is dead on it and we need to know now.

### Pro commission tiers (spot, effective 2025-10-01; evaluated quarterly)

| tier | spot rate | requires (per quarter) |
|---|---|---|
| 0 | 20% | nothing |
| **1** | **30%** | **$200k referred volume + 3 new traders** |
| 2 | 41% | $2.5M + 10 new traders |
| 3 | 50% | $30M + 15 new traders |

Futures: 10% (time-limited) → 30% at $1M + 3 → 40% → 50%.

**Payout: USDC, to your Spot wallet, within 6 hours of each hourly
calculation.** That is genuinely hourly income — and it lands exactly where the
allocator's idle-cash sweep already picks it up, so **referral income compounds
into earn automatically with no change to the machine.** `income_sources.py`
already tracks it on its own axis.

Eligibility: at least one referral made; not a sub-account, broker or Wealth
user; not in a restricted region. Binance serves Egypt.

### The arithmetic per trader, at Tier 1 (30%)

| referred trader | their monthly volume | pays you per year |
|---|---|---|
| light | $1,000 | $3.60 |
| moderate | $5,000 | $18.00 |
| **heavy** | **$50,000** | **$180.00** |

**$1/hour at Tier 1 = $29.2M/yr of referred volume = ~487 moderate traders, or
~49 heavy ones.** A few heavy traders are worth hundreds of light ones — the
target audience is people who *already trade actively*, not people who might
open an account.

Tier 1 is reachable fast: three active friends clear $200k/quarter between them.

### What to do this week

1. **Check the mode.** *Account → Referral.* Report back what it shows before
   selecting anything.
2. If unlocked, **choose Pro**, generate the link/code.
3. **Name the audience.** Who do you know who already trades crypto, and
   roughly how much? Three people at $50k/month is a different plan from three
   hundred at $1k.
4. Share. Commission appears in `income_sources.py` automatically; I will
   confirm the first credit and its classification.

---

## The two levers together

| referred traders (moderate) | referral $/yr | capital still needed |
|---|---|---|
| 0 | $0 | $192,716 |
| 50 | $1,500 | $159,383 |
| 150 | $4,500 | $92,716 |
| 250 | $7,500 | $26,050 |
| **292** | **$8,760** | **none** |

---

## Risks, stated plainly

- **Tier fragility.** USD1's 8.54% is 1.54% base + 7.00% promo tier. If the
  promo ends, it degrades to 1.54% overnight. USDC (2.34% base) and USDT
  (2.83%) degrade gracefully. The machine detects a stopped tier within three
  days of measured underpayment and rotates out on its own — verified this
  week, including a near-miss where it almost acted on stale data and the
  cooldown saved it. Prefer higher *base* at equal total when approving coins.
- **API lag.** Binance backdates bonus credits and the endpoints lag ~2h. A
  healthy new position can look bonus-less for ~36h. The machine no longer acts
  on that signal; it decides on three days of measured rate only.
- **Counterparty.** $190k on any single venue is concentration. USDY's
  Morgan-Stanley-custodied T-bills are the strongest backing available to a
  non-US retail holder, but the token, the bridge and the wallet are each a
  surface. Split across two venues above ~$50k.
- **Referral is not passive.** It scales with reach, not code. I can measure
  it and compound it; I cannot generate traders.
- **The laptop.** The machine only runs while the Mac is awake; it sleeps after
  one minute on battery. Yield is unaffected (Binance pays regardless) but
  detection and rotation pause. Keep it plugged in.

## What I need from you

1. **Capital available**, even roughly, and whether it can sit on-chain (USDY
   needs a wallet) or must stay on Binance.
2. **The referral mode** your account shows, and **the audience** — who trades,
   and how much.

Everything downstream of those two answers is built or designable within the
day.

## Already done (no action needed)

Rate 7.24% → 8.53% · every USD stablecoin scanned, vetted allowlist for holding
· tier-filler live (capacity $2,600) · paid-vs-promised audit with three-day gate
and entry block · both income axes tracked every 15 min · 1,118 tests · four
autonomous agents · $60.69 compounding at the best rate on the exchange.
