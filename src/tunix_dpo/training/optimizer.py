"""
Optimizer factories.

Provides two parallel implementations of the same schedule:

- ``make_optimizer`` (JAX/optax) — used by JAX-based reference implementations
  and by the unit tests.
- ``make_optimizer_torch`` (PyTorch) — used by the actual training loop in
  ``train.py`` since Gemma 3 ships as a PyTorch model.

Both use AdamW with linear warmup → cosine decay and global-norm gradient
clipping. The math is identical; only the framework differs.
"""

from __future__ import annotations

from tunix_dpo.training.config import TrainingConfig


# ── JAX / Optax variant ───────────────────────────────────────────────────────

def make_lr_schedule(cfg: TrainingConfig, total_steps: int):
    """Optax cosine schedule with linear warmup."""
    import optax

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


def make_optimizer(cfg: TrainingConfig, total_steps: int):
    """JAX/Optax AdamW with global-norm clipping and cosine warmup schedule."""
    import optax

    schedule = make_lr_schedule(cfg, total_steps)
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9, b2=0.95, eps=1e-8,
            weight_decay=cfg.weight_decay,
        ),
    )


# ── PyTorch variant ───────────────────────────────────────────────────────────

def make_optimizer_torch(model, cfg: TrainingConfig, total_steps: int):
    """PyTorch AdamW + LambdaLR (warmup → cosine). Returns (optimizer, scheduler).

    Gradient clipping is applied manually in the train loop because torch
    expects ``clip_grad_norm_`` between backward and step.
    """
    import math

    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )

    if cfg.lr_schedule == "constant":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
        return optimizer, scheduler

    warmup_steps = max(1, cfg.warmup_steps)
    cosine_steps = max(1, total_steps - warmup_steps)
    min_lr_ratio = 0.1  # mirrors optax's alpha=0.1

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / cosine_steps
        progress = min(max(progress, 0.0), 1.0)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos_factor

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler
