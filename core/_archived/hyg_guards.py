from decimal import Decimal, ROUND_DOWN

# ========================
# Structured Exceptions
# ========================

class MinNotionalViolation(Exception):
    """Raised when order notional < exchange MIN_NOTIONAL."""
    ...

class FeeSafetyViolation(Exception):
    """Raised when order quote value is below fee-safety floor."""
    ...

class MinEntryViolation(Exception):
    """Raised when order quote value < configured MIN_ENTRY_USDT."""
    ...

class IntegrityError(Exception):
    """Raised when invalid qty/price/filters passed to hygiene guards."""
    ...


# ========================
# Helpers
# ========================

def round_step(qty: Decimal, step: Decimal) -> Decimal:
    """Round quantity DOWN to exchange step (LOT_SIZE.stepSize)."""
    if step is None or step == 0:
        return qty
    n = (qty / step).to_integral_value(rounding=ROUND_DOWN)
    return n * step


def _get_cfg_val(config, key: str, default):
    try:
        return getattr(config, key)
    except Exception:
        return default


# ========================
# Enforcers
# ========================

def enforce_min_entry_quote(quote_value: float, min_entry_usdt: float):
    if float(quote_value) < float(min_entry_usdt):
        raise MinEntryViolation(f"quote {quote_value} < min-entry {min_entry_usdt}")


def enforce_fee_safety(quote_value: float, fee_safety_multiplier: float):
    # Prevent trades so tiny that fees dominate (e.g., 3x taker fee buffer)
    if float(quote_value) <= 0:
        raise FeeSafetyViolation("non-positive quote value")
    floor = 0.5 * float(fee_safety_multiplier)  # heuristic floor
    if quote_value < floor:
        raise FeeSafetyViolation(f"quote {quote_value} < safety floor {floor}")


# ========================
# Unified Guard
# ========================

def compute_tradable_qty(
    *,
    qty: float,
    price: float,
    step: float = None,
    min_qty: float = None,
    min_notional: float = None,
    config=None
) -> float:
    """
    Return a safe, rounded quantity or raise a structured violation.

    - qty: proposed base asset amount
    - price: latest price
    - step: LOT_SIZE.stepSize
    - min_qty: LOT_SIZE.minQty (optional)
    - min_notional: MIN_NOTIONAL.minNotional (optional)
    - config: provides HYG_MIN_ENTRY_USDT and HYG_FEE_SAFETY_MULTIPLIER
    """
    if qty is None or price is None or float(price) <= 0:
        raise IntegrityError("invalid qty/price input to compute_tradable_qty")

    q = Decimal(str(qty))
    st = Decimal(str(step)) if step else None
    q = round_step(q, st) if st else q
    qf = float(q)

    # Enforce minQty if provided
    if min_qty is not None and qf < float(min_qty):
        q = Decimal(str(min_qty))
        if st:
            q = round_step(q, st)
        qf = float(q)

    # Compute quote value
    quote_val = qf * float(price)

    # Config thresholds
    min_entry = float(_get_cfg_val(config, "HYG_MIN_ENTRY_USDT", 20.0))
    fee_mult = float(_get_cfg_val(config, "HYG_FEE_SAFETY_MULTIPLIER", 3.0))

    # Enforce floors
    enforce_min_entry_quote(quote_val, min_entry)
    if min_notional is not None and quote_val < float(min_notional):
        raise MinNotionalViolation(f"notional {quote_val} < min_notional {min_notional}")
    enforce_fee_safety(quote_val, fee_mult)

    return float(q)


# ========================
# Adapter
# ========================

def validate_order_request(symbol: str, side: str, quantity: float, price: float, filters: dict, config=None):
    """
    Adapter that validates an order using normalized filters, raises on failure.
    Filters dict expected shape:
      {"LOT_SIZE": {"stepSize","minQty","maxQty"},
       "MIN_NOTIONAL": {"minNotional"}}
    """
    lot = (filters or {}).get("LOT_SIZE", {})
    min_notional = (filters or {}).get("MIN_NOTIONAL", {}).get("minNotional")
    step = lot.get("stepSize")
    min_qty = lot.get("minQty")
    _ = compute_tradable_qty(
        qty=quantity,
        price=price,
        step=step,
        min_qty=min_qty,
        min_notional=min_notional,
        config=config,
    )
    # Returns None if ok; raises otherwise
    return None
