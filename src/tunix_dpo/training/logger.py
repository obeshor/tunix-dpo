"""
Training trajectory logger — TensorBoard + JSONL sidecar.

Uses ``tensorboardX`` for the TensorBoard summary writer (a lightweight
dependency that does NOT require PyTorch). Falls back to JSONL-only logging
if neither tensorboardX nor torch is installed.

The JSONL sidecar is always written and is the canonical record — anything
that goes to TensorBoard also goes to JSONL.
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
        self._jsonl = open(self.jsonl_path, "a", buffering=1)  # line-buffered

        self.tb = self._open_tensorboard()
        self.start_time = time.time()

    def _open_tensorboard(self):
        """Try tensorboardX first (no torch dep), then torch.utils.tensorboard."""
        try:
            from tensorboardX import SummaryWriter
            return SummaryWriter(logdir=str(self.log_dir))
        except ImportError:
            pass
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            logger.info("No TensorBoard writer available; logging JSONL only.")
            return None

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        """Record one step of metrics to JSONL (always) and TensorBoard (if available)."""
        elapsed = time.time() - self.start_time
        row: dict[str, Any] = {"step": step, "elapsed_s": round(elapsed, 2)}
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
