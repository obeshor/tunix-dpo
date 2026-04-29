"""Unit tests for tunix_dpo.serving.metrics."""

from tunix_dpo.serving.metrics import Metrics


class TestMetrics:
    def test_initial_state(self) -> None:
        m = Metrics()
        assert m.requests_total   == 0
        assert m.tokens_generated == 0
        assert m.errors_total     == 0

    def test_record(self) -> None:
        m = Metrics()
        m.record(tokens=50, latency_ms=200.0)
        assert m.requests_total   == 1
        assert m.tokens_generated == 50
        assert m.avg_latency_ms   == 200.0

    def test_multiple_records(self) -> None:
        m = Metrics()
        m.record(100, 100.0)
        m.record(200, 300.0)
        assert m.requests_total   == 2
        assert m.tokens_generated == 300
        assert m.avg_latency_ms   == 200.0  # (100 + 300) / 2

    def test_record_error(self) -> None:
        m = Metrics()
        m.record_error()
        m.record_error()
        assert m.errors_total == 2

    def test_next_id_increments(self) -> None:
        m  = Metrics()
        id1 = m.next_id()
        id2 = m.next_id()
        assert id1 != id2
        assert id1 == "req-00000001"
        assert id2 == "req-00000002"

    def test_prometheus_text_format(self) -> None:
        m = Metrics()
        m.record(10, 100.0)
        text = m.prometheus_text()
        assert "tunix_requests_total 1" in text
        assert "tunix_tokens_generated_total 10" in text
        assert "tunix_errors_total 0" in text
        assert "# TYPE" in text
