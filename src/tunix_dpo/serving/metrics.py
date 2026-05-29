"""In-process metrics for the inference server (pure data, Prometheus output)."""

from __future__ import annotations

import time


class Metrics:
    """Lightweight rolling metrics for inference traffic."""

    def __init__(self) -> None:
        self.requests_total = 0
        self.errors_total = 0
        self.tokens_generated = 0
        self.total_latency_ms = 0.0
        self.start_time = time.time()
        self._counter = 0

    def record(self, tokens: int, latency_ms: float) -> None:
        self.requests_total += 1
        self.tokens_generated += int(tokens)
        self.total_latency_ms += float(latency_ms)

    def record_error(self) -> None:
        self.errors_total += 1

    @property
    def avg_latency_ms(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.total_latency_ms / self.requests_total

    @property
    def uptime_s(self) -> float:
        return time.time() - self.start_time

    def next_id(self) -> str:
        self._counter += 1
        return f"req-{self._counter:08d}"

    def prometheus_text(self) -> str:
        return (
            f"# HELP tunix_requests_total Total inference requests served\n"
            f"# TYPE tunix_requests_total counter\n"
            f"tunix_requests_total {self.requests_total}\n"
            f"# HELP tunix_errors_total Total inference errors\n"
            f"# TYPE tunix_errors_total counter\n"
            f"tunix_errors_total {self.errors_total}\n"
            f"# HELP tunix_tokens_generated_total Total tokens generated\n"
            f"# TYPE tunix_tokens_generated_total counter\n"
            f"tunix_tokens_generated_total {self.tokens_generated}\n"
            f"# HELP tunix_avg_latency_ms Average request latency in ms\n"
            f"# TYPE tunix_avg_latency_ms gauge\n"
            f"tunix_avg_latency_ms {self.avg_latency_ms:.3f}\n"
            f"# HELP tunix_uptime_seconds Server uptime in seconds\n"
            f"# TYPE tunix_uptime_seconds counter\n"
            f"tunix_uptime_seconds {self.uptime_s:.1f}\n"
        )
