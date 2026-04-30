"""
Pure JAX loss functions — no I/O, no side effects.

Both functions are jit-compatible and pmap-safe.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def log_probs_from_logits(
    logits: jnp.ndarray,  # [batch, seq_len, vocab_size]
    labels: jnp.ndarray,  # [batch, seq_len]  — -100 = masked
) -> jnp.ndarray:  # [batch]
    """Sum of per-token log-probabilities over non-masked positions."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_lp = jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
    mask = (labels != -100).astype(jnp.float32)
    return (token_lp * mask).sum(axis=-1)


def dpo_loss(
    policy_chosen_logps: jnp.ndarray,  # [batch]
    policy_rejected_logps: jnp.ndarray,  # [batch]
    ref_chosen_logps: jnp.ndarray,  # [batch]
    ref_rejected_logps: jnp.ndarray,  # [batch]
    beta: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Direct Preference Optimisation loss (Rafailov et al., 2023).

    ℒ = -𝔼 log σ( β·[log π(y_w|x) - log π_ref(y_w|x)]
                  - β·[log π(y_l|x) - log π_ref(y_l|x)] )

    Returns
    -------
    loss             : scalar
    chosen_rewards   : [batch]  implicit reward on chosen responses
    rejected_rewards : [batch]  implicit reward on rejected responses
    """
    chosen_ratios = policy_chosen_logps - ref_chosen_logps
    rejected_ratios = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_ratios - rejected_ratios)
    loss = -jax.nn.log_sigmoid(logits).mean()
    return loss, chosen_ratios, rejected_ratios


def sft_loss(
    logits: jnp.ndarray,  # [batch, seq_len, vocab_size]
    labels: jnp.ndarray,  # [batch, seq_len]
) -> jnp.ndarray:
    """Causal next-token prediction cross-entropy.

    Logged alongside DPO loss every step to compare training trajectories.
    """
    mask = (labels != -100).astype(jnp.float32)
    n_tok = mask.sum()
    xe = optax.softmax_cross_entropy_with_integer_labels(
        logits, jnp.where(labels == -100, 0, labels)
    )
    return (xe * mask).sum() / jnp.maximum(n_tok, 1.0)
