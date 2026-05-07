from __future__ import annotations

from core_engine.native.cadence_scheduler import CadenceConfig, CadenceScheduler


def test_cadence_scheduler_initially_due() -> None:
    sched = CadenceScheduler()
    assert sched.is_due("decision", now=100.0) is True


def test_cadence_scheduler_respects_interval() -> None:
    sched = CadenceScheduler(CadenceConfig(decision_sec=30.0))
    sched.mark("decision", now=100.0)
    assert sched.is_due("decision", now=120.0) is False
    assert sched.is_due("decision", now=130.1) is True


def test_cadence_scheduler_tracks_multiple_lanes_independently() -> None:
    sched = CadenceScheduler(CadenceConfig(decision_sec=30.0, scenario_sec=60.0))
    sched.mark("decision", now=100.0)
    sched.mark("scenario", now=100.0)
    assert sched.is_due("decision", now=131.0) is True
    assert sched.is_due("scenario", now=131.0) is False
