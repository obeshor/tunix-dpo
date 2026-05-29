"""
Training configuration dataclasses (pure — no logic, no I/O).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_name: str = "google/gemma-3-1b-it"
    """HuggingFace model ID. Gemma 3 1B IT is text-only and uses
    ``AutoModelForCausalLM`` / ``Gemma3ForCausalLM`` for loading."""

    max_seq_len: int = 1024
    """Sequence length cap during training. Gemma 3 1B supports up to 32K
    tokens but DPO preference pairs rarely need more than ~1K."""

    dtype: str = "bfloat16"
    """TPU v5e native datatype. Recommended for Gemma 3 — quality may
    degrade with float16."""


@dataclass
class TrainingConfig:
    beta: float = 0.1
    """KL coefficient. Typical range 0.05–0.5. Higher = stronger alignment
    but greater risk of forgetting base capabilities."""

    learning_rate: float = 1.0e-5
    lr_schedule: str = "cosine"
    warmup_steps: int = 100
    num_epochs: int = 1

    per_device_batch_size: int = 8
    """Per-chip batch. With 8 chips on a v5e-8 this gives a global batch of 64.
    Gemma 3 1B is ~2 GB in bfloat16 — comfortably fits."""

    gradient_accumulation_steps: int = 1

    sft_loss_weight: float = 0.0
    """0 = pure DPO. >0 blends in standard SFT cross-entropy on the chosen
    response. Keep 0 for the canonical paper."""

    log_sft_loss: bool = True
    """Always compute SFT loss for logging, even when not used in the gradient."""

    max_grad_norm: float = 1.0
    weight_decay: float = 0.01


@dataclass
class DataConfig:
    data_dir: str = "./data/dpo"
    train_file: str = "train.jsonl"
    eval_file: str = "test.jsonl"
    num_workers: int = 4


@dataclass
class InfraConfig:
    gcs_bucket: str = "gs://your-project-tunix-checkpoints"
    checkpoint_dir: str = "checkpoints/dpo_v5e_run"
    checkpoint_every_steps: int = 500
    log_dir: str = "logs/dpo_v5e_run"
    log_every_steps: int = 10
    eval_every_steps: int = 250
    seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
