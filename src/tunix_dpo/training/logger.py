"""
Training trajectory logger.

Writes every metric to both TensorBoard (via tensorboardX) and a local
``metrics.jsonl`` for offline analysis. The JSONL file is consumed by
``plot_trajectories.py`` and the Phase 3 benchmark dashboard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

try:
    from tensorboardX import SummaryWriter  # type: ignore[import]
    _HAS_TB = True
except ImportError:
    _HAS_TB = False
    log.warning("tensorboardX not found — writing metrics to JSONL only.")


class TrajectoryLogger:
    """Logs scalar metrics to TensorBoard and a JSONL sidecar file.

    Parameters
    ----------
    log_dir:
        Directory where ``events.out.tfevents.*`` and ``metrics.jsonl``
        are written.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self._dir  = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = self._dir / "metrics.jsonl"
        self._writer = SummaryWriter(str(self._dir)) if _HAS_TB else None

    def log(
        self,
        step: int,
        metrics: dict[str, float],
        prefix: str = "train",
    ) -> None:
        """Write a dict of scalar metrics at the given step.

        Parameters
        ----------
        step:
            Global training step.
        metrics:
            ``{metric_name: value}`` — all values must be Python floats.
        prefix:
            ``"train"`` or ``"eval"``.
        """
        record = {"step": step, "prefix": prefix, **metrics}
        with open(self._jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(f"{prefix}/{k}", float(v), global_step=step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    # Context-manager support
    def __enter__(self) -> "TrajectoryLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
