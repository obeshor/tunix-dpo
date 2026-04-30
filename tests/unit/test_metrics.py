"""Unit tests for tunix_dpo.serving.metrics."""

from __future__ import annotations

from tunix_dpo.serving.metrics import Metrics


class TestMetrics:
    def test_initial_state(self) -> None:
        m = Metrics()
        assert m.requests_total == 0
        assert m.errors_total == 0
        assert m.tokens_generated == 0
        assert m.avg_latency_ms == 0.0

    def test_record_increments_counters(self) -> None:
        m = Metrics()
        m.record(tokens=50, latency_ms=120.0)
        m.record(tokens=80, latency_ms=180.0)
        assert m.requests_total == 2
        assert m.tokens_generated == 130
        assert abs(m.avg_latency_ms - 150.0) < 1e-6

    def test_record_error(self) -> None:
        m = Metrics()
        m.record_error()
        m.record_error()
        assert m.errors_total == 2

    def test_next_id_is_unique_and_monotonic(self) -> None:
        m = Metrics()
        ids = [m.next_id() for _ in range(5)]
        assert len(set(ids)) == 5
        assert ids[0] < ids[-1]

    def test_prometheus_text_includes_all_metrics(self) -> None:
        m = Metrics()
        m.record(tokens=10, latency_ms=50.0)
        text = m.prometheus_text()
        for k in ("tunix_requests_total", "tunix_tokens_generated_total",
                  "tunix_avg_latency_ms", "tunix_uptime_seconds"):
            assert k in text
