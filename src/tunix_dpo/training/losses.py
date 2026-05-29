"""
Pure JAX loss functions: DPO and SFT.

No I/O, no side effects. The training loop calls these inside a ``jax.jit``
or ``jax.pmap`` to compile and shard across the 8 v5e chips.

KEY INVARIANT
-------------
Logits at position ``t`` predict the token at position ``t+1``. Every loss
function in this module shifts inputs accordingly:

    log_p(token_t)  =  log_softmax(logits[t-1])[token_t]

so we drop the last logit and the first token before gathering.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def log_probs_from_logits(logits, labels, mask):
    """Sum log p(labels | context) over the response tokens only.

    Implements the canonical shift: logits at position t-1 predict token t.

    Parameters
    ----------
    logits : (B, T, V) float
        Raw logits from the model over the full sequence.
    labels : (B, T) int
        Token IDs at each position. labels[t] is the gold token *at* t.
    mask : (B, T) int
        1 over response tokens, 0 over prompt and padding. Mask is over the
        *labels* axis — i.e. it identifies which gold tokens to score.

    Returns
    -------
    (B,) float — total log-probability the model assigns to the response.
    """
    # logits[:, :-1] predicts labels[:, 1:]. The mask must match the labels.
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    shifted_mask = mask[:, 1:]

    log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)
    selected = jnp.take_along_axis(log_probs, shifted_labels[..., None], axis=-1).squeeze(-1)

    return (selected * shifted_mask).sum(axis=-1)


def dpo_loss(
    policy_logits_chosen,
    policy_logits_rejected,
    ref_logits_chosen,
    ref_logits_rejected,
    labels_chosen,
    labels_rejected,
    mask_chosen,
    mask_rejected,
    beta: float = 0.1,
):
    """Compute the DPO loss for one batch of preference pairs.

    The DPO objective is::

        ℒ = −E [ log σ( β · ((logπ_θ(y_w|x) − logπ_ref(y_w|x))
                            − (logπ_θ(y_l|x) − logπ_ref(y_l|x))) ) ]

    Returns
    -------
    loss : float
        Scalar mean DPO loss over the batch.
    metrics : dict
        - reward_acc:    fraction of pairs where the model already prefers
                         chosen over rejected
        - reward_margin: mean of (chosen_logratio − rejected_logratio)
    """
    pol_chosen = log_probs_from_logits(policy_logits_chosen, labels_chosen, mask_chosen)
    pol_rejected = log_probs_from_logits(policy_logits_rejected, labels_rejected, mask_rejected)

    ref_chosen = jax.lax.stop_gradient(
        log_probs_from_logits(ref_logits_chosen, labels_chosen, mask_chosen)
    )
    ref_rejected = jax.lax.stop_gradient(
        log_probs_from_logits(ref_logits_rejected, labels_rejected, mask_rejected)
    )

    chosen_logratio = pol_chosen - ref_chosen
    rejected_logratio = pol_rejected - ref_rejected

    logits = beta * (chosen_logratio - rejected_logratio)
    losses = -jax.nn.log_sigmoid(logits)

    metrics = {
        "reward_margin": (chosen_logratio - rejected_logratio).mean(),
        "reward_acc": (chosen_logratio > rejected_logratio).mean().astype(jnp.float32),
        "policy_chosen_logp": pol_chosen.mean(),
        "policy_rejected_logp": pol_rejected.mean(),
    }
    return losses.mean(), metrics


def sft_loss(logits, labels, mask):
    """Standard SFT (next-token cross-entropy) loss over the response tokens.

    Applies the same shift as ``log_probs_from_logits``: logits[t-1] predicts
    labels[t], so we drop the last logit and the first label.
    """
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    shifted_mask = mask[:, 1:]

    log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)
    selected = jnp.take_along_axis(log_probs, shifted_labels[..., None], axis=-1).squeeze(-1)
    n_tokens = shifted_mask.sum() + 1e-8
    return -(selected * shifted_mask).sum() / n_tokens
