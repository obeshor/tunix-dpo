"""
Optimizer factory — gradient clipping → AdamW with cosine warmup.
"""

from __future__ import annotations

import optax

from tunix_dpo.training.config import TrainingConfig


def make_lr_schedule(cfg: TrainingConfig, total_steps: int) -> optax.Schedule:
    """Cosine schedule with linear warmup. Falls back to constant if requested."""
    if cfg.lr_schedule == "constant":
        return optax.constant_schedule(cfg.learning_rate)

    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=cfg.learning_rate,
        transition_steps=cfg.warmup_steps,
    )
    cosine_steps = max(1, total_steps - cfg.warmup_steps)
    cosine = optax.cosine_decay_schedule(
        init_value=cfg.learning_rate,
        decay_steps=cosine_steps,
        alpha=0.1,
    )
    return optax.join_schedules([warmup, cosine], boundaries=[cfg.warmup_steps])


def make_optimizer(cfg: TrainingConfig, total_steps: int) -> optax.GradientTransformation:
    """AdamW with global-norm clipping and cosine warmup schedule."""
    schedule = make_lr_schedule(cfg, total_steps)
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=cfg.weight_decay,
        ),
    )
