"""
Dynamic rule overrides loader.

This module reads `automation/proposed_rules.json` (if present) and exposes
get_required_conf_override(symbol) -> float | None which returns a suggested
required_conf for the given symbol. The file is reloaded on each call to keep
behavior dynamic without restarting the orchestrator.
"""
import json
import os
import time
from functools import lru_cache

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'proposed_rules.json')
_CACHE_TTL = 30.0  # seconds
_last_load_ts = 0.0
_last_payload = {}

# Control helpers: evaluate toggles dynamically per-call.
# This allows (a) updating environment variables in-process, and (b)
# toggling via a simple local control file `automation/.apply_proposed_rules`
# without restarting the orchestrator.
_CONTROL_FILE = os.path.join(os.path.dirname(__file__), '.apply_proposed_rules')


def _apply_enabled():
    """Return True if applying proposals is enabled.

    Priority: control file (if present) -> environment variable -> default True
    """
    # Control file presence allows operators to toggle without restarting the process.
    try:
        if os.path.exists(_CONTROL_FILE):
            with open(_CONTROL_FILE, 'r') as f:
                val = f.read().strip().lower()
            return val in ('1', 'true', 'yes', 'on')
    except Exception:
        # ignore and fall back to env var
        pass
    return os.environ.get('APPLY_PROPOSED_RULES', 'true').lower() in ('1', 'true', 'yes', 'on')


def _min_conf_floor():
    """Return configured minimum required_conf floor as a float (default 0.50)."""
    try:
        return float(os.environ.get('MIN_REQUIRED_CONF_FLOOR', '0.50'))
    except Exception:
        return 0.50


def _load(path=DEFAULT_PATH):
    global _last_load_ts, _last_payload
    now = time.time()
    if (now - _last_load_ts) < _CACHE_TTL and _last_payload:
        return _last_payload

    if not os.path.exists(path):
        _last_payload = {}
        _last_load_ts = now
        return {}
    try:
        with open(path, 'r') as f:
            payload = json.load(f)
        proposals = payload.get('proposals') if isinstance(payload, dict) else None
        if not proposals:
            _last_payload = {}
            _last_load_ts = now
            return {}
        mapping = {p['symbol']: p for p in proposals if 'symbol' in p}
        _last_payload = mapping
        _last_load_ts = now
        return mapping
    except Exception:
        _last_payload = {}
        _last_load_ts = now
        return {}


def get_required_conf_override(symbol):
    """Return suggested_required_conf for symbol or None."""
    if not _apply_enabled():
        return None
    try:
        proposals = _load()
        p = proposals.get(symbol)
        if not p:
            return None
        val = p.get('suggested_required_conf')
        if val is None:
            return None
        # safety cap: do not allow suggested conf below configured floor
        allow_below = bool(p.get('allow_below_floor', False))
        try:
            valf = float(val)
        except Exception:
            return None
        floor = _min_conf_floor()
        if valf < floor and not allow_below:
            # do not apply override below floor
            return None
        return valf
    except Exception:
        return None
