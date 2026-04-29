"""
Request telemetry — no FastAPI dependency.

Keeps a running tally of requests, tokens, errors, and latency.
Exposes a ``prometheus_text()`` method for the /metrics endpoint.
"""

from __future__ import annotations

import time


class Metrics:
    """Thread-safe (single-process async) telemetry store."""

    def __init__(self) -> None:
        self.requests_total   = 0
        self.tokens_generated = 0
        self.errors_total     = 0
        self.latency_sum_ms   = 0.0
        self._req_id          = 0
        self._start           = time.time()

    def next_id(self) -> str:
        self._req_id += 1
        return f"req-{self._req_id:08d}"

    def record(self, tokens: int, latency_ms: float) -> None:
        self.requests_total   += 1
        self.tokens_generated += tokens
        self.latency_sum_ms   += latency_ms

    def record_error(self) -> None:
        self.errors_total += 1

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_sum_ms / max(self.requests_total, 1)

    @property
    def avg_tokens_per_sec(self) -> float:
        elapsed = max(time.time() - self._start, 1e-6)
        return self.tokens_generated / elapsed

    def prometheus_text(self) -> str:
        lines = [
            "# HELP tunix_requests_total Total inference requests",
            "# TYPE tunix_requests_total counter",
            f"tunix_requests_total {self.requests_total}",
            "# HELP tunix_tokens_generated_total Tokens generated",
            "# TYPE tunix_tokens_generated_total counter",
            f"tunix_tokens_generated_total {self.tokens_generated}",
            "# HELP tunix_errors_total Request errors",
            "# TYPE tunix_errors_total counter",
            f"tunix_errors_total {self.errors_total}",
            "# HELP tunix_avg_latency_ms Average latency ms",
            "# TYPE tunix_avg_latency_ms gauge",
            f"tunix_avg_latency_ms {self.avg_latency_ms:.2f}",
            "# HELP tunix_tokens_per_second Token throughput",
            "# TYPE tunix_tokens_per_second gauge",
            f"tunix_tokens_per_second {self.avg_tokens_per_sec:.2f}",
        ]
        return "\n".join(lines) + "\n"
