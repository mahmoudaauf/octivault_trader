"""Platform health: a job is judged on its PRODUCT, not on being loaded.

The failure mode this exists for: a StartInterval job that errors on EVERY run
reports exit=0 for its last attempt, stays loaded, appears in launchctl list —
and produces nothing. This project lost six weeks to that shape in July 2026
when macOS TCC silently denied three cron jobs disk access.
"""
from __future__ import annotations

import importlib
import time

import pytest

ph = importlib.import_module("platform_health")


def test_freshness_budget_is_derived_from_the_job_s_own_cadence(tmp_path, monkeypatch):
    """Hardcoding a threshold means a cadence change silently breaks the check —
    either passing a job running at the wrong rate, or failing a healthy one."""
    import plistlib
    monkeypatch.setattr(ph, "AGENTS_DIR", tmp_path)
    with open(tmp_path / "com.octivault.x.plist", "wb") as f:
        plistlib.dump({"Label": "com.octivault.x", "StartInterval": 900}, f)
    budget, how = ph.budget_minutes("com.octivault.x", 999)
    assert budget == pytest.approx(15 * ph.MISSED)     # 15m cadence x missed
    assert "15m" in how


def test_a_job_with_no_interval_falls_back_rather_than_passing_blindly(tmp_path, monkeypatch):
    monkeypatch.setattr(ph, "AGENTS_DIR", tmp_path)
    budget, how = ph.budget_minutes("com.octivault.missing", 42)
    assert budget == 42 and "default" in how


def test_environment_warning_noise_is_not_an_error(tmp_path, monkeypatch):
    """Every script here emits the urllib3/LibreSSL warning on every run.
    Surfacing it would mark all seven agents erroring forever — no signal."""
    monkeypatch.setattr(ph, "ROOT", tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "a.err").write_text(
        "/x/urllib3/__init__.py:35: NotOpenSSLWarning: ...\n  warnings.warn\n")
    assert ph.stderr_tail("logs/a.log", 999) is None


def test_a_stale_error_is_ignored_because_the_job_has_run_clean_since(tmp_path, monkeypatch):
    """dnyscan flagged on a TimeoutError that was 2 occurrences against 201
    completed scans. An old error on a job that has since succeeded is noise."""
    monkeypatch.setattr(ph, "ROOT", tmp_path)
    (tmp_path / "logs").mkdir()
    err = tmp_path / "logs" / "a.err"
    err.write_text("asyncio.exceptions.TimeoutError\n")
    old = time.time() - 60 * 60 * 24
    import os
    os.utime(err, (old, old))
    assert ph.stderr_tail("logs/a.log", 30) is None          # older than budget
    assert ph.stderr_tail("logs/a.log", 60 * 48) is not None  # inside a wide budget


def test_a_fresh_real_error_is_surfaced(tmp_path, monkeypatch):
    monkeypatch.setattr(ph, "ROOT", tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "a.err").write_text("APIError -2008 Invalid Api-Key\n")
    assert "Invalid Api-Key" in ph.stderr_tail("logs/a.log", 60)


def test_a_loaded_job_producing_nothing_is_DOWN_not_OK(tmp_path, monkeypatch):
    """The whole point: loaded is nearly meaningless."""
    monkeypatch.setattr(ph, "ROOT", tmp_path)
    monkeypatch.setattr(ph, "AGENTS_DIR", tmp_path)
    monkeypatch.setattr(ph, "OUTPUTS", {"com.octivault.x": ("logs/never.log", 30)})
    monkeypatch.setattr(ph, "loaded", lambda: {"com.octivault.x": "0"})
    r = ph.check()[0]
    assert r["state"] == "DOWN" and "has ever been written" in r["why"]


def test_a_silent_job_is_STALE(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(ph, "ROOT", tmp_path)
    monkeypatch.setattr(ph, "AGENTS_DIR", tmp_path)
    monkeypatch.setattr(ph, "OUTPUTS", {"com.octivault.x": ("logs/x.log", 30)})
    monkeypatch.setattr(ph, "loaded", lambda: {"com.octivault.x": "0"})
    (tmp_path / "logs").mkdir()
    log = tmp_path / "logs" / "x.log"
    log.write_text("hi")
    old = time.time() - 60 * 60 * 5
    os.utime(log, (old, old))
    r = ph.check()[0]
    assert r["state"] == "STALE" and "silent" in r["why"]
