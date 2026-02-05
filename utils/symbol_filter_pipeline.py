from typing import List, Dict, Any, Optional

def filter_symbols(
    raw_symbols: List[str],
    market_stats: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    blacklist: Optional[List[str]] = None,
    top_n: int = 40,
    min_volume: float = 500_000,
    usdt_only: bool = True,
    volatility_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Filter and rank symbols. Returns a dict keyed by symbol so callers can do `.items()`.

    Args:
        raw_symbols: universe of symbols (e.g., from exchange info)
        market_stats: map symbol -> { volume, price, spread, ... } (24h stats)
        config: config dict (used for disallow lists, etc.)
        blacklist: symbols to exclude (strings or dicts with a 'symbol' key)
        top_n: max count to return
        min_volume: minimum 24h *quote* volume to qualify
        usdt_only: keep only quote-asset = USDT (simple suffix check)
        volatility_scores: optional map symbol -> score (higher ranks first)

    Returns:
        Dict[str, Dict] of { symbol: {symbol, source, vol24h_quote, price, score?} }
    """
    # Normalize blacklist -> set of strings
    bl_set = set()
    if blacklist:
        for item in blacklist:
            if isinstance(item, dict):
                s = item.get("symbol")
                if isinstance(s, str):
                    bl_set.add(s.strip().upper())
            elif isinstance(item, str):
                bl_set.add(item.strip().upper())

    # Disallowed leveraged/ETP suffixes (configurable)
    disallow_suffixes = tuple(
        config.get(
            "DISALLOW_SUFFIXES",
            config.get("disallow_suffixes", ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"))
        )
    )

    # Helper to compute quote volume robustly
    def _quote_volume(stats: Dict[str, Any]) -> float:
        base_vol = float(stats.get("volume") or 0.0)
        last_price = float(stats.get("lastPrice") or stats.get("price") or 0.0)
        quote_vol = float(stats.get("quoteVolume") or (base_vol * last_price))
        return quote_vol

    # Filter quickly
    candidates: List[str] = []
    for sym in raw_symbols:
        if not isinstance(sym, str):
            continue
        s = sym.strip().upper()
        if s in bl_set:
            continue
        if usdt_only and not s.endswith("USDT"):
            continue
        # Exclude leveraged/ETP-style tickers that also end with USDT
        if any(s.endswith(suf) for suf in disallow_suffixes):
            continue

        stats = market_stats.get(s) or {}

        # Skip non-trading / non-spot if such flags exist in stats
        status = (stats.get("status") or "").strip().upper()
        if status and status != "TRADING":
            continue
        if stats.get("isSpotTradingAllowed") is False:
            continue

        # Volume filter uses *quote* volume
        if _quote_volume(stats) < float(min_volume):
            continue

        candidates.append(s)

    # Rank by volatility_scores if provided, else by 24h quote volume desc
    if volatility_scores:
        candidates.sort(
            key=lambda sym: float(volatility_scores.get(sym, 0.0)),
            reverse=True,
        )
    else:
        candidates.sort(
            key=lambda sym: _quote_volume(market_stats.get(sym) or {}),
            reverse=True,
        )

    # Build result dict (slice top_n)
    out: Dict[str, Dict[str, Any]] = {}
    limit = max(0, int(top_n))
    for s in candidates[:limit]:
        stats = market_stats.get(s) or {}
        vol_q = _quote_volume(stats)
        price = float(stats.get("lastPrice") or stats.get("price") or 0.0)
        out[s] = {
            "symbol": s,
            "source": "filtered_pipeline",
            "vol24h_quote": vol_q,
            "price": price,
        }
        if volatility_scores:
            out[s]["score"] = float(volatility_scores.get(s, 0.0))

    return out
