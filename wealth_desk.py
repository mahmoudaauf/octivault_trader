#!/usr/bin/env python3
"""The wealth desk — measures PURCHASING POWER, not dollars.

WHY THIS EXISTS, AND WHY IT MATTERS MORE THAN THE TRADING DESK
---------------------------------------------------------------
Every other instrument in this project measures dollars. That is the wrong unit
for this principal. They live and spend in Egypt, so the only number that means
anything is what the portfolio buys in EGP after Egyptian inflation.

Reframing to that unit produces an uncomfortable result. This office spent
weeks lifting the yield from 7.24% to 8.60% — real, permanent, and the largest
gain the account ever made. Egypt's CPI is 14.5%. So in purchasing-power terms
the portfolio is LOSING about 6% a year, and would be even at double the rate.

Worse, the yield is second-order. A 10 percentage-point move in USD/EGP is
worth roughly seven times the entire 1.36pp of yield improvement. Currency
exposure is the first-order decision here and yield is the second, which is the
opposite of how this engagement has been run.

THE COMPARISON THIS DESK EXISTS TO SHOW
----------------------------------------
Egyptian bank certificates pay 17.25% in EGP, tax-exempt (NBE Platinum, Banque
Misr Al Qimma, April 2026). Against 14.5% CPI that is about +2.75% real. Our
USD yield is 8.60%, whose real value depends ENTIRELY on the EGP path:

    EGP devalues 15%  →  +10.4% real   (USD wins by 7.6pp)
    EGP devalues  5%  →   −0.5% real   (CD wins by 3.2pp)
    EGP flat          →   −5.9% real   (CD wins by 8.6pp)
    EGP gains 6.7%    →  −13.2% real   (CD wins by 15.9pp — and this is what
                                        actually happened in 2025)

WHAT THIS DESK WILL NOT DO
--------------------------
Predict the currency. Nobody can, and this project's entire record is a
demonstration that prediction does not work. It presents the scenarios and
names which one you are implicitly betting on by holding what you hold, because
an unexamined FX position is still a position.

It also will not recommend moving to EGP. That decision carries bank risk,
capital-control risk and reversibility questions this desk cannot price, and
the principal chose crypto for reasons this desk was not told. It measures; the
principal decides.

Read-only. One public FX read; no credentials, no orders, no money.

Usage:  python3 wealth_desk.py [--nav 60.31] [--cpi 14.5] [--cd 17.25]
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

# Egypt CPI, year-on-year, August 2026 (eased from 14.9% in July).
DEFAULT_CPI = float(os.getenv("EGY_CPI_PCT", "14.5"))
# NBE Platinum / Banque Misr Al Qimma monthly certificates, April 2026 onward.
# Tax-exempt under current Egyptian law, which is why no tax haircut is applied.
DEFAULT_CD = float(os.getenv("EGY_CD_PCT", "17.25"))
FALLBACK_FX = 51.70


def fx_rate() -> tuple[float, str]:
    """USDT/EGP, from the P2P book this project already samples.

    There is no USDTEGP spot pair on Binance — EGP is a P2P-only market, which
    is itself the reason the spread this office found exists. So the rate comes
    from p2p_spread_monitor's own history: the mid of the best buy and best
    sell, which is the price a person actually transacts near.
    """
    try:
        rows = [json.loads(l) for l in open(
            os.getenv("P2P_STATE", "logs/p2p_spread_history.jsonl")) if l.strip()]
        if rows:
            r = rows[-1]
            return (r["buy"] + r["sell"]) / 2.0, f"P2P mid, {len(rows)} samples"
    except Exception:
        pass
    return FALLBACK_FX, "FALLBACK — no P2P history yet"


def real_return(usd_yield: float, egp_move: float, cpi: float) -> float:
    """Purchasing-power return in EGP.

    `egp_move` is the EGP's move against USD: negative means the pound weakened,
    which RAISES the EGP value of a dollar asset. Compounded rather than added,
    because a 15% devaluation on top of an 8.6% yield is multiplicative.
    """
    nominal_egp = (1 + usd_yield) * (1 - egp_move) - 1
    return nominal_egp - cpi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", type=float, default=60.31)
    ap.add_argument("--usd-yield", type=float, default=8.60, help="our verified rate, %%")
    ap.add_argument("--cpi", type=float, default=DEFAULT_CPI)
    ap.add_argument("--cd", type=float, default=DEFAULT_CD)
    a = ap.parse_args()
    y, cpi, cd = a.usd_yield / 100, a.cpi / 100, a.cd / 100
    fx, src = fx_rate()

    print("=" * 74)
    print("WEALTH DESK — measured in what it BUYS, not what it is worth in dollars")
    print("=" * 74)
    print(f"\n  portfolio    ${a.nav:,.2f}  =  {a.nav*fx:,.0f} EGP    (USDT/EGP {fx:.2f}, {src})")
    print(f"  our yield    {a.usd_yield:.2f}% in USD")
    print(f"  Egypt CPI    {a.cpi:.1f}%  ← the hurdle every asset must clear")
    print(f"  EGP bank CD  {a.cd:.2f}% tax-exempt  →  {(cd-cpi)*100:+.2f}% real\n")

    print("  THE UNCOMFORTABLE NUMBER")
    print(f"    At {a.usd_yield:.2f}% USD with the pound flat, purchasing power changes by")
    print(f"    {real_return(y,0.0,cpi)*100:+.1f}% a year. The yield is not the problem — "
          f"{a.cpi:.1f}% inflation is.")
    print(f"    Even doubling the rate to {a.usd_yield*2:.1f}% would still be "
          f"{real_return(y*2,0.0,cpi)*100:+.1f}% real.\n")

    print("  WHAT YOU ARE IMPLICITLY BETTING ON BY HOLDING DOLLARS")
    print(f"    {'EGP vs USD':>18}{'EGP nominal':>14}{'real':>10}{'vs the CD':>12}")
    rows = [(-0.15, "devalues 15%"), (-0.10, "devalues 10%"), (-0.05, "devalues 5%"),
            (0.0, "flat"), (0.067, "gains 6.7% (2025)")]
    for mv, lbl in rows:
        nom = (1 + y) * (1 - mv) - 1
        r = nom - cpi
        print(f"    {lbl:>18}{nom*100:>13.1f}%{r*100:>9.1f}%{(r-(cd-cpi))*100:>11.1f}pp")
    # The break-even is the honest headline: how far must the pound fall for the
    # dollar position to merely MATCH a bank certificate the principal could buy.
    # egp_move is negative when the pound weakens, so the break-even move is
    # negative too; report its magnitude or the sentence reads backwards.
    be = abs(1 - (1 + cd) / (1 + y))
    print(f"\n    BREAK-EVEN: the pound must WEAKEN {be*100:.1f}% a year for our "
          f"{a.usd_yield:.2f}% USD")
    print(f"    to merely MATCH a {a.cd:.2f}% Egyptian certificate. Weaken less than "
          f"that — or strengthen — and the CD wins.")

    print("\n  SCALE CHECK — where the office has been spending its effort")
    print(f"    yield improvement won so far   1.36pp  (7.24% → {a.usd_yield:.2f}%)")
    print(f"    a 10pp currency move is worth  {10/1.36:.0f}x that")
    print("    Currency is the first-order decision. Yield is the second.")

    print("\n  WHAT THIS DESK DOES NOT DO")
    print("    Predict the pound — nobody can, and this project's whole record shows")
    print("    prediction fails. Nor recommend moving to EGP: that carries bank risk,")
    print("    capital-control risk and reversibility this desk cannot price.")
    print("    It names the bet you are already making. The choice stays yours.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
