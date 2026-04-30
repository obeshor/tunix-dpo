"""Unit tests for tunix_dpo.training.losses (skipped if JAX absent)."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from tunix_dpo.training.losses import dpo_loss, log_probs_from_logits, sft_loss


def _fake_logits(B: int = 2, T: int = 8, V: int = 16, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (B, T, V))


def _fake_labels(B: int = 2, T: int = 8, V: int = 16, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    return jax.random.randint(key, (B, T), 0, V)


def _ones_mask(B: int = 2, T: int = 8):
    return jnp.ones((B, T), dtype=jnp.int32)


class TestLogProbsFromLogits:
    def test_shape(self) -> None:
        out = log_probs_from_logits(_fake_logits(), _fake_labels(), _ones_mask())
        assert out.shape == (2,)


class TestDpoLoss:
    def test_returns_scalar_loss(self) -> None:
        loss, metrics = dpo_loss(
            _fake_logits(seed=1), _fake_logits(seed=2),
            _fake_logits(seed=1), _fake_logits(seed=2),
            _fake_labels(seed=1), _fake_labels(seed=2),
            _ones_mask(), _ones_mask(),
            beta=0.1,
        )
        assert loss.ndim == 0
        assert "reward_acc" in metrics
        assert "reward_margin" in metrics

    def test_zero_when_policy_equals_reference(self) -> None:
        """If policy == reference, DPO logits collapse to 0 → loss ≈ log 2."""
        logits_c = _fake_logits(seed=42)
        logits_r = _fake_logits(seed=99)
        labels_c = _fake_labels(seed=42)
        labels_r = _fake_labels(seed=99)
        mask = _ones_mask()

        loss, _ = dpo_loss(
            logits_c, logits_r, logits_c, logits_r,
            labels_c, labels_r, mask, mask, beta=0.1,
        )
        assert abs(float(loss) - 0.6931) < 1e-3


class TestSftLoss:
    def test_positive(self) -> None:
        l = sft_loss(_fake_logits(), _fake_labels(), _ones_mask())
        assert float(l) > 0
