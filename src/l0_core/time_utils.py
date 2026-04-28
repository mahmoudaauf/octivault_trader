"""Layer-0 timestamp / time utilities.

Pure functions with no upstream layer dependencies. Used by L1+ for
coercion of mixed-format timestamps (epoch seconds, ISO 8601, etc.).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

_logger = logging.getLogger(__name__)


def parse_timestamp(val: Any, default_ts: float = 0.0) -> float:
    """Coerce a timestamp-like value into epoch seconds.

    Accepts:
      * ``None``               → ``default_ts``
      * ``int`` / ``float``    → coerced to ``float``
      * ISO-8601 ``str``       → parsed (``Z`` suffix supported)
      * anything else          → ``default_ts``

    Never raises; logs at DEBUG on parse failure and returns
    ``default_ts`` instead.
    """
    if val is None:
        return default_ts
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            iso = val[:-1] + "+00:00" if val.endswith("Z") else val
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception as e:  # broad — never raise from a coercion helper
            _logger.debug("parse_timestamp: failed on %r (%s); returning default", val, e)
            return default_ts
    return default_ts
