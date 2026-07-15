#!/usr/bin/env python3
"""
Binance news-sentiment backtest ("semantic detection from news").

Tests three hypotheses using REAL historical Binance announcements (pulled
directly from Binance's own announcement archive, not a third-party feed)
paired with real historical spot price data:

  H1 (New Listing, catalogId=48):   does a new-listing announcement predict
     positive forward returns for the listed symbol?
  H2 (Delisting, catalogId=161):    does a delisting announcement predict
     negative forward returns for the affected symbol before removal?
  H3 (Latest Binance News, catalogId=49): does a deterministic lexicon
     sentiment score (VADER + a small crypto-domain keyword lexicon) on a
     general headline mentioning a specific symbol predict its forward
     return?

No LLM calls anywhere -- VADER + a small crypto keyword lexicon, zero cost,
fully deterministic and reproducible. Same real-out-of-sample discipline as
funding_carry_backtest.py: thresholds fixed in advance, chronological
in-sample/out-of-sample split reported separately, small-n explicitly
flagged rather than oversold as edge.

Usage:  python3 news_sentiment_backtest.py
Env:    NEWS_MAX_LISTING (2000) NEWS_MAX_DELISTING (413, i.e. all)
        NEWS_MAX_LATEST (2000) NEWS_HORIZONS_H ("1,4,24,72")
        NEWS_HORIZONS_H1 ("0.25,0.5,1,4,24,72") -- H1 gets extra short
        horizons to test for a "listing pump" separate from the 1h+ effect.
        NEWS_MIN_TRADES (30) NEWS_SENTIMENT_THRESHOLD (0.3)
        NEWS_ROUND_TRIP_COST_PCT (0.24) -- subtracted from every trade's
        direction-adjusted return, matching funding_carry_backtest.py's
        cost-accounting convention.
        NEWS_STRESSED_COST_PCT (1.0) -- a second, higher-cost scenario
        reported alongside the base one for H2 specifically, since delisted
        coins are frequently thin/illiquid and the "normal" cost assumption
        may understate real slippage.
        NEWS_REFRESH_CACHE (false)
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BAPI_URL = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
CATALOG_NEW_LISTING = 48
CATALOG_DELISTING = 161
CATALOG_LATEST_NEWS = 49

CACHE_DIR = "data/news_archive"

MAX_LISTING = int(os.getenv("NEWS_MAX_LISTING", "1000"))
MAX_DELISTING = int(os.getenv("NEWS_MAX_DELISTING", "413"))
MAX_LATEST = int(os.getenv("NEWS_MAX_LATEST", "2000"))
HORIZONS_H = [float(x) for x in os.getenv("NEWS_HORIZONS_H", "1,4,24,72").split(",")]
HORIZONS_H1 = [float(x) for x in os.getenv("NEWS_HORIZONS_H1", "0.25,0.5,1,4,24,72").split(",")]
MIN_TRADES = int(os.getenv("NEWS_MIN_TRADES", "30"))
SENTIMENT_THRESHOLD = float(os.getenv("NEWS_SENTIMENT_THRESHOLD", "0.3"))
ROUND_TRIP_COST_PCT = float(os.getenv("NEWS_ROUND_TRIP_COST_PCT", "0.24"))
STRESSED_COST_PCT = float(os.getenv("NEWS_STRESSED_COST_PCT", "1.0"))
REFRESH_CACHE = os.getenv("NEWS_REFRESH_CACHE", "false").lower() in ("1", "true", "yes")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}

# Crypto-domain keyword adjustments layered on top of VADER's generic-English
# compound score -- VADER alone doesn't know "delisting" or "exploit" carry
# crypto-specific weight beyond their mild generic-English sentiment.
CRYPTO_LEXICON_BOOST = {
    "hack": -0.6, "hacked": -0.6, "exploit": -0.6, "exploited": -0.6,
    "vulnerability": -0.5, "lawsuit": -0.5, "sec": -0.3, "halt": -0.4,
    "halted": -0.4, "suspend": -0.4, "suspended": -0.4, "delist": -0.5,
    "delisting": -0.5, "removal": -0.3, "monitoring tag": -0.3,
    "partnership": 0.3, "launch": 0.2, "launches": 0.2, "listing": 0.3,
    "airdrop": 0.3, "integration": 0.2, "upgrade": 0.15,
    # Expanded pass (round 2) -- more crypto-specific terms VADER's generic
    # English lexicon doesn't weight correctly on its own.
    "outage": -0.4, "downtime": -0.3, "compromise": -0.5, "compromised": -0.5,
    "breach": -0.5, "restrict": -0.3, "restricted": -0.3, "ban": -0.5,
    "banned": -0.5, "fraud": -0.6, "scam": -0.6, "rug pull": -0.7,
    "investigation": -0.4, "insolvent": -0.6, "insolvency": -0.6,
    "freeze": -0.4, "frozen": -0.4, "etf": 0.3, "approval": 0.3,
    "approved": 0.3, "adoption": 0.2, "surge": 0.2, "record": 0.15,
    "milestone": 0.2, "expansion": 0.15, "collaborate": 0.2,
    "collaboration": 0.2, "reward": 0.15, "bonus": 0.15, "promotion": 0.1,
    "compensate": 0.2, "compensation": 0.2, "reimburse": 0.2,
    "reimbursement": 0.2, "insurance fund": 0.2, "recover": 0.15,
    "recovered": 0.2, "resume": 0.2, "resumed": 0.25, "stabilize": 0.1,
}

# Sub-type keyword flags -- captured per H1/H2 sample so the report can show
# whether a futures or margin market was implicated (a proxy for whether
# shorting/exiting fast was actually possible at announcement time, since we
# have no direct historical margin/futures-availability API to query).
def classify_subtype(title: str) -> str:
    lower = title.lower()
    if "futures" in lower or "perpetual" in lower:
        return "futures"
    if "margin" in lower:
        return "margin"
    return "spot"

# All-caps tokens that show up in titles but are not tickers.
TICKER_STOPWORDS = {
    "USD", "USDT", "BUSD", "THE", "AND", "FOR", "NEW", "API", "ATH", "APR",
    "ADD", "ADDS", "ADDING", "WILL", "NOTICE", "REMOVAL", "SPOT", "MARGIN",
    "LOAN", "ON", "OF", "TO", "IS", "BINANCE", "FUTURES", "EXCHANGE",
    "LAUNCH", "LAUNCHES", "TRADING", "PAIR", "PAIRS", "PAIR(S)", "LISTING",
    "DELIST", "DELISTING", "PERPETUAL", "CONTRACT", "CONTRACTS", "UPDATE",
    "UPDATES", "UPDATED", "EARN", "ALPHA", "EIO", "CEASE", "SUPPORT",
    "STOCKS", "TICK", "SIZE", "CAPITAL", "CONNECT", "PORTFOLIO",
}
TICKER_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")


def extract_tickers(title: str) -> list[str]:
    tokens = TICKER_RE.findall(title)
    out: list[str] = []
    for t in tokens:
        base = t[:-4] if t.endswith("USDT") else t
        if base in TICKER_STOPWORDS or len(base) < 2:
            continue
        if base not in out:
            out.append(base)
    return out


def fetch_catalog_articles(catalog_id: int, max_articles: int) -> list[dict]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"catalog_{catalog_id}.json")
    if os.path.exists(cache_path) and not REFRESH_CACHE:
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) >= max_articles:
            print(f"[news] catalog {catalog_id}: using {len(cached)} cached articles")
            return cached[:max_articles]

    page_size = 20
    articles: list[dict] = []
    page = 1
    total = None
    while len(articles) < max_articles:
        try:
            resp = requests.get(
                BAPI_URL,
                params={"type": 1, "catalogId": catalog_id, "pageNo": page, "pageSize": page_size},
                headers=HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            cat = data["data"]["catalogs"][0]
            total = cat["total"]
            batch = cat["articles"]
        except Exception as e:
            print(f"[news] catalog {catalog_id} page {page} failed: {str(e)[:100]}")
            break
        if not batch:
            break
        articles.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
        time.sleep(0.25)  # polite pacing -- undocumented endpoint, don't hammer it

    print(f"[news] catalog {catalog_id}: fetched {len(articles)}/{total} articles")
    with open(cache_path, "w") as f:
        json.dump(articles, f)
    return articles[:max_articles]


def score_headline(title: str, analyzer: SentimentIntensityAnalyzer) -> float:
    base = analyzer.polarity_scores(title)["compound"]
    lower = title.lower()
    boost = sum(w for k, w in CRYPTO_LEXICON_BOOST.items() if k in lower)
    return max(-1.0, min(1.0, base + boost))


async def forward_returns(
    client, symbol: str, ts_ms: int, horizons_h: list[float], interval: str = "1h"
) -> dict | None:
    """Returns {"returns": {horizon_h: pct_return}, "pre_volume_usdt": float|None}.
    pre_volume_usdt is the average hourly quote-asset volume in the ~24h before
    ts_ms -- a liquidity proxy: thin pre-announcement volume is a real warning
    sign that fast execution (shorting/exiting) may not be feasible at size.
    None if no usable price data at all."""
    pair = f"{symbol}USDT"
    max_h = max(horizons_h)
    start = ts_ms - 25 * 3600_000  # extra lookback for the pre-volume diagnostic
    end = ts_ms + int(max_h * 3600_000) + 3600_000
    try:
        klines = await client.get_historical_klines(
            pair, interval, start_str=str(start), end_str=str(end)
        )
    except Exception:
        return None
    if not klines or len(klines) < 2:
        return None

    pre_vols = [float(k[7]) for k in klines if k[0] <= ts_ms]  # quoteAssetVolume
    if pre_vols:
        hours_covered = len(pre_vols) * _interval_ms(interval) / 3600_000
        pre_volume_usdt = sum(pre_vols) / hours_covered  # avg $ volume / hour
    else:
        pre_volume_usdt = None

    anchor = None
    for k in klines:
        if k[0] <= ts_ms:
            anchor = float(k[4])
        else:
            break
    if anchor is None:
        anchor = float(klines[0][4])
    if anchor <= 0:
        return None

    returns = {}
    for h in horizons_h:
        target_ts = ts_ms + int(h * 3600_000)
        candidate = None
        for k in klines:
            if k[0] <= target_ts:
                candidate = float(k[4])
            else:
                break
        if candidate is not None:
            returns[h] = (candidate - anchor) / anchor * 100.0
    if not returns:
        return None
    return {"returns": returns, "pre_volume_usdt": pre_volume_usdt}


def _interval_ms(interval: str) -> int:
    unit = interval[-1]
    n = int(interval[:-1])
    return n * {"m": 60_000, "h": 3600_000, "d": 86_400_000}[unit]


def split_in_out_sample(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    ordered = sorted(samples, key=lambda s: s["ts"])
    cut = int(len(ordered) * 0.7)
    return ordered[:cut], ordered[cut:]


def report_hypothesis(name: str, samples: list[dict], horizon_h: float, cost_pct: float = ROUND_TRIP_COST_PCT) -> None:
    print("\n" + "=" * 72)
    print(f"{name} — horizon {horizon_h:.2f}h  (cost model: {cost_pct:.2f}% round-trip)")
    print("=" * 72)
    if not samples:
        print("  No samples with resolvable price data.")
        return

    def stats(rows: list[dict]) -> tuple[int, float, float]:
        rets = [r["returns"][horizon_h] * r["direction"] - cost_pct
                for r in rows if horizon_h in r["returns"]]
        n = len(rets)
        if n == 0:
            return 0, 0.0, 0.0
        wins = sum(1 for x in rets if x > 0)
        avg = sum(rets) / n
        return n, avg, wins / n * 100.0

    in_sample, out_sample = split_in_out_sample(samples)
    n_in, avg_in, wr_in = stats(in_sample)
    n_out, avg_out, wr_out = stats(out_sample)
    n_all, avg_all, wr_all = stats(samples)

    print(f"  All:          n={n_all:<5} avg_net_ret={avg_all:+.3f}%  win-rate={wr_all:.0f}%")
    print(f"  In-sample:    n={n_in:<5} avg={avg_in:+.3f}%  win-rate={wr_in:.0f}%  (first 70% chronologically)")
    print(f"  Out-of-sample n={n_out:<5} avg={avg_out:+.3f}%  win-rate={wr_out:.0f}%  (last 30%, holdout)")

    print("-" * 72)
    if n_out < MIN_TRADES:
        print(f"  VERDICT: ⏳ INCONCLUSIVE — only {n_out}/{MIN_TRADES} out-of-sample samples.")
    elif avg_out > 0 and wr_out > 50:
        print(f"  VERDICT: ✅ EDGE CANDIDATE (out-of-sample, net of {cost_pct:.2f}% cost) — "
              f"avg {avg_out:+.3f}%, {wr_out:.0f}% win.")
    else:
        print(f"  VERDICT: ❌ NO EDGE (out-of-sample, net of {cost_pct:.2f}% cost) — "
              f"avg {avg_out:+.3f}%, {wr_out:.0f}% win.")
    print("-" * 72)


def report_execution_feasibility(name: str, samples: list[dict]) -> None:
    """Diagnostics for whether the signal is actually tradeable in practice --
    not a return number, a liquidity/market-availability sanity check."""
    print("\n" + "=" * 72)
    print(f"{name} — EXECUTION FEASIBILITY DIAGNOSTICS")
    print("=" * 72)
    if not samples:
        print("  No samples.")
        return

    subtypes: dict[str, int] = {}
    for s in samples:
        subtypes[s["subtype"]] = subtypes.get(s["subtype"], 0) + 1
    total = len(samples)
    print("  Announcement sub-type (proxy for whether a shortable/exitable market")
    print("  existed at announcement time -- title mentions futures/margin explicitly):")
    for st, cnt in sorted(subtypes.items(), key=lambda x: -x[1]):
        print(f"    {st:8} {cnt:>4}  ({cnt/total*100:.0f}%)")

    vols = sorted(s["pre_volume_usdt"] for s in samples if s.get("pre_volume_usdt") is not None)
    if vols:
        p10 = vols[int(len(vols) * 0.10)]
        p50 = vols[int(len(vols) * 0.50)]
        p90 = vols[int(len(vols) * 0.90)]
        print(f"\n  Pre-announcement avg hourly volume (USDT), n={len(vols)}:")
        print(f"    p10={p10:,.0f}   median={p50:,.0f}   p90={p90:,.0f}")
        thin = sum(1 for v in vols if v < 50_000)
        print(f"    {thin}/{len(vols)} ({thin/len(vols)*100:.0f}%) averaged <$50k/hour "
              f"pre-announcement -- likely too thin to execute at size without material slippage.")
    print("-" * 72)


async def build_samples(
    client, articles: list[dict], direction_fn, valid_syms: set[str],
    horizons_h: list[float] = HORIZONS_H, interval: str = "1h",
) -> list[dict]:
    samples: list[dict] = []
    seen_calls: dict[tuple, dict] = {}
    total = len(articles)
    for i, art in enumerate(articles, 1):
        title = art.get("title", "")
        ts_ms = art.get("releaseDate")
        if not title or not ts_ms:
            continue
        direction = direction_fn(title)
        if direction == 0:
            continue
        tickers = extract_tickers(title)
        resolved = [t for t in tickers if t in valid_syms]
        if not resolved:
            continue
        subtype = classify_subtype(title)
        for sym in resolved:
            call_key = (sym, ts_ms // (3600_000 * 6), interval)  # dedupe within a 6h bucket
            if call_key in seen_calls:
                result = seen_calls[call_key]
            else:
                result = await forward_returns(client, sym, ts_ms, horizons_h, interval=interval)
                seen_calls[call_key] = result
                await asyncio.sleep(0.12)  # polite pacing on Binance REST
            if result:
                samples.append({
                    "symbol": sym, "ts": ts_ms, "direction": direction,
                    "returns": result["returns"], "pre_volume_usdt": result["pre_volume_usdt"],
                    "subtype": subtype,
                })
        if i % 50 == 0:
            print(f"    ...{i}/{total} articles processed, {len(samples)} samples so far")
    return samples


async def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dotenv import load_dotenv

    load_dotenv()
    from binance import AsyncClient

    key = os.getenv("BINANCE_API_KEY") or "x"
    sec = os.getenv("BINANCE_API_SECRET") or "x"
    client = await AsyncClient.create(key, sec)
    analyzer = SentimentIntensityAnalyzer()

    try:
        print("Fetching current + historical Binance spot symbol universe...")
        info = await client.get_exchange_info()
        valid_syms = {
            s["baseAsset"] for s in info["symbols"] if s.get("quoteAsset") == "USDT"
        }
        print(f"  {len(valid_syms)} currently-listed USDT base assets known.\n")

        print("Fetching Binance announcement archive (catalogs 48/161/49)...")
        listing_articles = fetch_catalog_articles(CATALOG_NEW_LISTING, MAX_LISTING)
        delisting_articles = fetch_catalog_articles(CATALOG_DELISTING, MAX_DELISTING)
        latest_articles = fetch_catalog_articles(CATALOG_LATEST_NEWS, MAX_LATEST)

        # H1: New Listing -> always a positive-direction event for the listed symbol.
        # H2: Delisting -> always negative-direction for the delisted symbol.
        # For H1/H2, valid_syms (currently-listed) under-covers historically-delisted
        # names; forward_returns() itself is the real gate -- if Binance no longer
        # serves klines for a since-delisted symbol we simply get no sample for it,
        # which is reported as reduced coverage, not silently faked.
        print("\nBuilding H1 (New Listing) samples -- 5m granularity to catch a short-horizon 'listing pump'...")
        h1_samples = await build_samples(
            client, listing_articles, lambda t: 1,
            valid_syms | _all_tickers(listing_articles),
            horizons_h=HORIZONS_H1, interval="5m",
        )

        print("\nBuilding H2 (Delisting) samples...")
        h2_samples = await build_samples(client, delisting_articles, lambda t: -1, valid_syms | _all_tickers(delisting_articles))

        print("\nBuilding H3 (Latest Binance News, sentiment-scored) samples...")

        def h3_direction(title: str) -> int:
            score = score_headline(title, analyzer)
            if score >= SENTIMENT_THRESHOLD:
                return 1
            if score <= -SENTIMENT_THRESHOLD:
                return -1
            return 0

        h3_samples = await build_samples(client, latest_articles, h3_direction, valid_syms)

    finally:
        await client.close_connection()

    report_execution_feasibility("H1: NEW LISTING", h1_samples)
    for h in HORIZONS_H1:
        report_hypothesis("H1: NEW LISTING -> forward return", h1_samples, h)

    report_execution_feasibility("H2: DELISTING", h2_samples)
    for h in HORIZONS_H:
        report_hypothesis("H2: DELISTING -> forward return", h2_samples, h)
    print("\n  ~~ H2 under a STRESSED cost assumption (thin/illiquid delisted-coin slippage) ~~")
    for h in HORIZONS_H:
        report_hypothesis("H2: DELISTING -> forward return [STRESSED COST]", h2_samples, h, cost_pct=STRESSED_COST_PCT)

    for h in HORIZONS_H:
        report_hypothesis("H3: GENERAL NEWS SENTIMENT -> forward return", h3_samples, h)


def _all_tickers(articles: list[dict]) -> set[str]:
    """H1/H2 concern symbols that may since be delisted (or, for H1, may not
    yet have existed in the CURRENT exchangeInfo snapshot at all if later
    delisted) -- widen the accepted-ticker set to anything the titles
    themselves name, and let forward_returns()'s real API call be the actual
    gate on whether usable price data exists."""
    out: set[str] = set()
    for art in articles:
        out.update(extract_tickers(art.get("title", "")))
    return out


if __name__ == "__main__":
    asyncio.run(main())
