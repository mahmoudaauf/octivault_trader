"""L7 — Observability: read-only, failure-contained."""
from src.l0_core.layer_contracts import L7ObservabilityContract
from tests.layers.fakes import FakeMetrics, FakePortfolio


def test_l7_contract_validates_required_fields():
    c = L7ObservabilityContract()
    assert c.validate_output({
        "metrics_emitted": 5, "alerts_emitted": 1, "last_scrape_ts": 0.0,
    })
    assert not c.validate_output({"metrics_emitted": 5})


def test_l7_metrics_failure_does_not_break_business_logic():
    """If a metric emit raises, L4-style code must still complete its work."""
    metrics = FakeMetrics(raise_on="gauge")
    p = FakePortfolio(cash=10_000.0)

    # Simulate L4 doing real work and "trying" to emit a metric
    tok = p.reserve("BTCUSDT", 4_000.0, "buy")
    try:
        metrics.gauge("reserve.qty", 4_000.0)
    except RuntimeError:
        pass        # L4 swallows L7 failures by contract
    assert p.apply_fill(tok, "BTCUSDT", 0.1, 40_000.0)
    fills = [e for e in p.journal if e["event"] == "FILL"]
    assert len(fills) == 1


def test_l7_alert_bus_preserves_order():
    metrics = FakeMetrics()
    metrics.emit("INFO", "L4", "first", {})
    metrics.emit("WARN", "L3", "second", {})
    metrics.emit("CRIT", "L1", "third", {})
    assert [a["msg"] for a in metrics.alerts] == ["first", "second", "third"]
