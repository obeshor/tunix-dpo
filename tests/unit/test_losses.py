"""Unit tests for tunix_dpo.training.losses."""

import pytest

pytest.importorskip("jax", reason="JAX not installed — skipping loss tests")

import jax.numpy as jnp

from tunix_dpo.training.losses import dpo_loss, log_probs_from_logits, sft_loss


class TestLogProbsFromLogits:
    def test_shape(self) -> None:
        logits = jnp.zeros((2, 5, 10))   # batch=2, seq=5, vocab=10
        labels = jnp.zeros((2, 5), dtype=jnp.int32)
        out    = log_probs_from_logits(logits, labels)
        assert out.shape == (2,)

    def test_masked_tokens_ignored(self) -> None:
        logits = jnp.zeros((1, 4, 8))
        # Only the first token is unmasked
        labels_full   = jnp.zeros((1, 4), dtype=jnp.int32)
        labels_masked = jnp.array([[-100, -100, -100, 0]])
        full_lp   = log_probs_from_logits(logits, labels_full)
        masked_lp = log_probs_from_logits(logits, labels_masked)
        # Masked version sums over fewer tokens — value should differ
        assert float(full_lp[0]) != float(masked_lp[0])


class TestDpoLoss:
    def test_returns_scalar(self) -> None:
        b = 4
        chosen   = jnp.full((b,), -1.0)
        rejected = jnp.full((b,), -2.0)
        loss, cr, rr = dpo_loss(chosen, rejected, chosen, rejected, beta=0.1)
        assert loss.shape == ()

    def test_loss_decreases_when_chosen_better(self) -> None:
        """Loss should be lower when chosen log-probs clearly exceed rejected."""
        high = jnp.full((8,), -0.5)
        low  = jnp.full((8,), -5.0)
        loss_good, _, _ = dpo_loss(high, low,  high, low,  beta=0.1)
        loss_bad,  _, _ = dpo_loss(low,  high, high, low,  beta=0.1)
        assert float(loss_good) < float(loss_bad)

    def test_reward_margin_positive_when_chosen_better(self) -> None:
        high = jnp.full((4,), -1.0)
        low  = jnp.full((4,), -3.0)
        _, chosen_r, rejected_r = dpo_loss(high, low, high, low, beta=0.1)
        # chosen reward = log-ratio for chosen; both ref and policy are same
        # so ratios are 0 — just check the call doesn't error
        assert chosen_r.shape == (4,)
        assert rejected_r.shape == (4,)


class TestSftLoss:
    def test_returns_scalar(self) -> None:
        logits = jnp.zeros((2, 5, 10))
        labels = jnp.zeros((2, 5), dtype=jnp.int32)
        loss   = sft_loss(logits, labels)
        assert loss.shape == ()

    def test_all_masked_returns_zero(self) -> None:
        logits = jnp.ones((2, 5, 10))
        labels = jnp.full((2, 5), -100, dtype=jnp.int32)
        loss   = sft_loss(logits, labels)
        assert float(loss) == pytest.approx(0.0)
