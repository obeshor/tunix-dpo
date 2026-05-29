"""Training package — DPO fine-tuning on TPU via JAX / Flax."""

from tunix_dpo.training.config import (
    Config,
    DataConfig,
    InfraConfig,
    ModelConfig,
    TrainingConfig,
)

__all__ = [
    "Config",
    "DataConfig",
    "InfraConfig",
    "ModelConfig",
    "TrainingConfig",
]

# Loss functions require JAX — guard so that data-only environments work.
try:
    from tunix_dpo.training.losses import dpo_loss, log_probs_from_logits, sft_loss

    __all__ += ["dpo_loss", "log_probs_from_logits", "sft_loss"]
except ImportError:
    pass
