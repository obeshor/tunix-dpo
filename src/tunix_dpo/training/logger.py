"""
Training trajectory logger — TensorBoard + JSONL sidecar.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TrajectoryLogger:
    """Dual-write metrics logger: TensorBoard for the dashboard, JSONL for replay."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self._jsonl = open(self.jsonl_path, "a", buffering=1)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            logger.info("tensorboard not installed; logging JSONL only")
            self.tb = None

        self.start_time = time.time()

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        elapsed = time.time() - self.start_time
        row = {"step": step, "elapsed_s": round(elapsed, 2)}
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = str(v)
        self._jsonl.write(json.dumps(row) + "\n")

        if self.tb is not None:
            for k, v in metrics.items():
                try:
                    self.tb.add_scalar(k, float(v), step)
                except (TypeError, ValueError):
                    pass

    def close(self) -> None:
        if self._jsonl:
            self._jsonl.close()
        if self.tb is not None:
            self.tb.close()
