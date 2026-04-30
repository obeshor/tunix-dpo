"""
Optimizer factory for DPO training.

Returns an optax GradientTransformation composed of:
  - gradient clipping
  - AdamW with weight decay
  - cosine / linear / constant LR schedule with warmup
"""

from __future__ import annotations

import optax
from tunix_dpo.training.config import TrainingConfig


def make_optimizer(cfg: TrainingConfig, total_steps: int) -> optax.GradientTransformation:
    """Build the optimizer from a TrainingConfig.

    Parameters
    ----------
    cfg:
        Training hyperparameters.
    total_steps:
        Total number of gradient update steps (used to build the LR schedule).
    """
    warmup = cfg.warmup_steps

    if cfg.lr_schedule == "cosine":
        schedule: optax.Schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.learning_rate,
            warmup_steps=warmup,
            decay_steps=total_steps,
            end_value=cfg.learning_rate * 0.1,
        )
    elif cfg.lr_schedule == "linear":
        ramp = optax.linear_schedule(0.0, cfg.learning_rate, warmup)
        decay = optax.linear_schedule(cfg.learning_rate, 0.0, total_steps - warmup)
        schedule = optax.join_schedules([ramp, decay], boundaries=[warmup])
    elif cfg.lr_schedule == "constant":
        schedule = cfg.learning_rate  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule!r}")

    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay),
    )
