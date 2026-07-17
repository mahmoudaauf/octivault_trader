# Funding-Carry Frontier Findings (2026-07-17)

**Verdict: positive-only funding carry — what `carry_paper_trader.py` actually
runs — has no viable operating point. The project's "+0.94%/trade validated
edge" is real, but ~98% of it lives in negative funding, which v1 structurally
cannot trade.**

Reproduce: `python3 carry_frontier_sweep.py` (caches funding to
`data/carry_funding_cache.json`; delete to refresh). This document exists
because `logs/` and `data/` are gitignored — the prior "26 trades / +0.90% /
73% win" liquidity-haircut claim was recorded only as memory prose, printed to
stdout, and was **unverifiable** when re-checked (zero hits across repo, git
history, docs, logs). Don't repeat that: results land here.

## 1. The headline number reproduces

| | trades | avg %/trade | win% |
|---|---|---|---|
| Recorded in memory (2026-07-15) | 944 | +0.9434% | 60% |
| Re-measured here (2026-07-17) | 929 | +0.9295% | 60% |

Same methodology (361 spot-hedgeable perps, `FUNDING_ENTRY_BPS=6`,
`abs(fr)` entry + `abs(fr)` collection). The number is real and reproducible.
**The problem is what it measures.**

## 2. Why funding is 75.7% positive but its EXTREMES are 98.4% negative

Across 179,732 funding windows (361 symbols, up to 166d each):

| | count | share |
|---|---|---|
| All windows, funding **positive** | 136,054 | **75.7%** |
| All windows, funding negative | 43,360 | 24.1% |
| **Windows crossing +6bps** (v1 CAN trade) | **75** | **1.6%** |
| **Windows crossing −6bps** (v1 CANNOT trade) | **4,475** | **98.4%** |

Funding is normally a mild positive contango (longs pay shorts). But *extreme*
funding is a panic signature — capitulation drives crowded shorts and deeply
negative rates. So the entry condition ("|funding| is extreme") and the
execution capability ("positive funding only") select almost-disjoint sets.

**v1 fishes in 1.6% of its own opportunity pond. This is a design
contradiction, not a drought and not bad luck.**

## 3. Decomposition (widest universe, $0M filter, entry 6bps)

| model | trades | trd/day | avg %/trade | win% |
|---|---|---|---|---|
| 1 validated (abs, both signs) | 929 | 9.421 | +0.9295 | 60% |
| 2 pos-only, abs collect | 33 | 0.272 | +0.2354 | 58% |
| 3 pos-only, signed (**= LIVE TODAY**) | 33 | 0.272 | **+0.0697** | **45%** |
| 4 pos-only, signed, exit-on-flip | 37 | 0.297 | +0.0930 | 41% |

- **1→2** (positive-only entry filter): **28x fewer trades** (929→33) and edge
  drops to a quarter. This is the 98.4%-negative-extremes effect above.
- **2→3** (committed-direction accounting): same trades, edge falls +0.235% →
  +0.070%, win 58% → **45%**. The old `abs()` collection counted mid-hold sign
  flips as *income* when a short-perp position is actually *paying*.
- **3→4** (proposed sign-flip exit): does **not** rescue it. +0.070% → +0.093%
  and win-rate gets *worse* (45%→41%). A hypothesised cheap fix that doesn't work.

At the daemon's **actual** config ($50M filter): **3 trades in ~166 days,
+0.02%/trade, 33% win.** Statistically nothing.

## 4. No entry threshold rescues positive-only

Positive-only + signed collection, cost 0.24% round-trip already subtracted:

| entry | trades ($0M) | trd/day | avg %/trade | win% |
|---|---|---|---|---|
| 1bps | 6,466 | 41.1 | **−0.2284** | **1%** |
| 2bps | 221 | 2.47 | −0.1327 | 12% |
| 3bps | 122 | 1.34 | −0.0786 | 19% |
| 4bps | 74 | 0.68 | −0.0370 | 23% |
| 5bps | 51 | 0.47 | +0.0138 | 33% |
| **6bps (live)** | **33** | **0.27** | **+0.0697** | **45%** |
| 8bps | 21 | 0.16 | +0.0635 | 43% |
| 10bps | 6 | 0.04 | +0.0130 | 50% |

**The frontier never delivers frequency AND edge simultaneously.** Mechanism:
over long holds positive and negative funding roughly *cancel*, so the trade
nets out to just the round-trip cost — which is precisely the −0.2284% seen at
1bps. Carry only pays when it captures extreme funding, and extreme funding is
98.4% negative. Win-rate never clears 50% at any threshold with meaningful n.

## 5. Two real measurement bugs found and fixed

- **`days_hist` was 2x wrong.** `funding_carry_backtest.py` hardcoded
  `days_hist = 1000 / 3.0` (333d) assuming `limit=1000` returns 1000 funding
  windows. Binance's `/fapi/v1/fundingRate` **caps at 500 rows** → a
  full-history symbol spans ~166d, and newer listings far less (HOMEUSDT ≈ 38d,
  median across the universe = 83d). Every trades/yr and %/yr figure printed
  under that assumption — including memory's "~+3.2%/yr/symbol" — was ~2x too
  low. Now measured per-symbol from real timestamps.
- **`max_funding` was an absolute value.** `carry_paper_trader.py` printed
  `max([abs(v) ...])`, so a −0.13% spike rendered as a tradeable-looking
  `0.130%`, and `positive_only` discarded it with **no log line at all**. 19 of
  80 live-armed polls (24%) "crossed" the threshold this way and every one was
  silently dropped. Now prints `best_tradeable=` alongside `abs_max=`, and logs
  each positive-only skip.

## 6. What this does and does not falsify

- **Falsified:** funding carry *as implemented* (positive-only, v1). No
  threshold makes it work.
- **NOT falsified:** funding carry *with both signs*. The +0.93%/trade, 60%
  win, n=929 result is real — but capturing it needs **negative-funding carry**
  (long perp + short spot), which requires spot-margin shorting. v1 explicitly
  excludes it, and the live API key currently reports `enableMargin: False`.
- **Caveat:** n=33 (pos-only, $0M) is small, and this is a single ~166d window
  in what looks like a persistently short-skewed regime. A different regime
  could shift the positive/negative extreme balance. The *structural* point —
  extremes are where the edge is, and extremes skew negative — is unlikely to
  invert, but the exact ratio is regime-dependent.
