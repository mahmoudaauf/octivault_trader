        if stats.get("isSpotTradingAllowed") is False:
            continue

        # Volume filter uses *quote* volume
        if _quote_volume(stats) < float(min_volume):
            continue
