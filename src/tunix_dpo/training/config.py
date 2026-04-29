"""
Training configuration dataclasses.

All fields have defaults so configs can be constructed programmatically
without a YAML file.  The YAML loader in train.py overlays these defaults
with values from disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_name:  str = "google/gemma-2b"
    max_seq_len: int = 512
    dtype:       str = "bfloat16"   # bfloat16 = TPU v5e native


@dataclass
class TrainingConfig:
    # DPO objective
    beta:                        float = 0.1      # KL coefficient
    learning_rate:               float = 1e-5
    lr_schedule:                 str   = "cosine" # cosine | linear | constant
    warmup_steps:                int   = 100
    num_epochs:                  int   = 1
    per_device_batch_size:       int   = 4
    gradient_accumulation_steps: int   = 2

    # SFT trajectory comparison
    sft_loss_weight: float = 0.0   # 0 = pure DPO; >0 = blended
    log_sft_loss:    bool  = True  # always log for comparison

    # Regularisation
    max_grad_norm: float = 1.0
    weight_decay:  float = 0.01


@dataclass
class DataConfig:
    data_dir:    str = "./data/dpo"
    train_file:  str = "train.jsonl"
    eval_file:   str = "test.jsonl"
    num_workers: int = 4


@dataclass
class InfraConfig:
    gcs_bucket:              str = "gs://your-project-tunix-checkpoints"
    checkpoint_dir:          str = "checkpoints/dpo_run"
    checkpoint_every_steps:  int = 500
    log_dir:                 str = "logs/dpo_run"
    log_every_steps:         int = 10
    eval_every_steps:        int = 250
    seed:                    int = 42


@dataclass
class Config:
    model:    ModelConfig    = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data:     DataConfig     = field(default_factory=DataConfig)
    infra:    InfraConfig    = field(default_factory=InfraConfig)
