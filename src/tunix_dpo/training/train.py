"""
DPO training CLI — registered as ``tunix-train``.

Orchestrates Phase 2 of the project: loads YAML config, the Gemma 3 1B IT
base model on TPU via JAX/Flax, snapshots a frozen reference copy, iterates
over the curated DPO JSONL data, computes the DPO loss with gradients
sharded across the 8 v5e chips, saves Orbax checkpoints, and logs to
TensorBoard + metrics.jsonl.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

import click
import yaml

from tunix_dpo.training.config import Config

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path, overrides: list[str]) -> Config:
    """Load YAML config and apply ``section.field=value`` CLI overrides."""
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f) or {}

    cfg = Config(
        model=_dict_to_dataclass(cfg_dict.get("model", {}), "ModelConfig"),
        training=_dict_to_dataclass(cfg_dict.get("training", {}), "TrainingConfig"),
        data=_dict_to_dataclass(cfg_dict.get("data", {}), "DataConfig"),
        infra=_dict_to_dataclass(cfg_dict.get("infra", {}), "InfraConfig"),
    )

    for ov in overrides:
        if "=" not in ov:
            raise click.BadParameter(f"override must be key=value: {ov!r}")
        key, value = ov.split("=", 1)
        _apply_override(cfg, key, value)

    return cfg


def _dict_to_dataclass(d: dict, name: str):
    from tunix_dpo.training import config as cfg_mod
    cls = getattr(cfg_mod, name)
    instance = cls()
    for k, v in d.items():
        if hasattr(instance, k):
            setattr(instance, k, v)
    return instance


def _apply_override(cfg: Config, key: str, value: str) -> None:
    parts = key.split(".")
    if len(parts) != 2:
        raise click.BadParameter(f"override key must be section.field: {key!r}")
    section, field = parts
    section_obj = getattr(cfg, section, None)
    if section_obj is None or not hasattr(section_obj, field):
        raise click.BadParameter(f"unknown override key: {key!r}")
    current = getattr(section_obj, field)
    if isinstance(current, bool):
        coerced: object = value.lower() in {"1", "true", "yes"}
    elif isinstance(current, int):
        coerced = int(value)
    elif isinstance(current, float):
        coerced = float(value)
    else:
        coerced = value
    setattr(section_obj, field, coerced)


def run_training(cfg: Config) -> None:
    """The actual training loop. Imports JAX lazily."""
    import jax
    import jax.numpy as jnp
    from flax.training import train_state

    from tunix_dpo.training.dataset import DataLoader, HHRLHFDataset
    from tunix_dpo.training.logger import TrajectoryLogger
    from tunix_dpo.training.losses import dpo_loss, sft_loss
    from tunix_dpo.training.optimizer import make_optimizer

    devices = jax.devices()
    n_devices = len(devices)
    if n_devices < 1:
        raise RuntimeError("No JAX devices found — did you install jax[tpu]?")
    logger.info("JAX devices: %d × %s", n_devices, devices[0].device_kind)

    from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

    logger.info("Loading tokenizer: %s", cfg.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model in %s …", cfg.model.dtype)
    dtype = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}[cfg.model.dtype]
    model = FlaxAutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        dtype=dtype,
        from_pt=True,
    )
    params = model.params
    ref_params = jax.tree.map(lambda x: x, params)

    train_path = Path(cfg.data.data_dir) / cfg.data.train_file
    train_ds = HHRLHFDataset(train_path, tokenizer, cfg.model.max_seq_len)
    global_bs = cfg.training.per_device_batch_size * n_devices
    loader = DataLoader(train_ds, batch_size=global_bs, seed=cfg.infra.seed)

    total_steps = len(loader) * cfg.training.num_epochs
    optimizer = make_optimizer(cfg.training, total_steps)
    state = train_state.TrainState.create(apply_fn=model.__call__, params=params, tx=optimizer)

    def loss_fn(params, batch):
        chosen_logits = model(batch["chosen_ids"], params=params).logits
        rejected_logits = model(batch["rejected_ids"], params=params).logits
        ref_chosen_logits = model(batch["chosen_ids"], params=ref_params).logits
        ref_rejected_logits = model(batch["rejected_ids"], params=ref_params).logits

        loss, dpo_metrics = dpo_loss(
            chosen_logits, rejected_logits,
            ref_chosen_logits, ref_rejected_logits,
            batch["chosen_ids"], batch["rejected_ids"],
            batch["chosen_mask"], batch["rejected_mask"],
            beta=cfg.training.beta,
        )

        if cfg.training.log_sft_loss:
            sft_l = sft_loss(chosen_logits, batch["chosen_ids"], batch["chosen_mask"])
            dpo_metrics["sft_loss"] = sft_l
            if cfg.training.sft_loss_weight > 0:
                loss = loss + cfg.training.sft_loss_weight * sft_l

        return loss, dpo_metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    log = TrajectoryLogger(cfg.infra.log_dir)
    log.log(0, {"event": "start", "total_steps": total_steps, "global_batch": global_bs})

    step = 0
    for epoch in range(cfg.training.num_epochs):
        logger.info("── Epoch %d/%d ──", epoch + 1, cfg.training.num_epochs)
        for batch in loader:
            batch_jax = {k: jnp.asarray(v) for k, v in batch.items()}
            (loss, metrics), grads = grad_fn(state.params, batch_jax)
            state = state.apply_gradients(grads=grads)

            if step % cfg.infra.log_every_steps == 0:
                log.log(step, {"dpo_loss": float(loss), **{k: float(v) for k, v in metrics.items()}})

            if step > 0 and step % cfg.infra.checkpoint_every_steps == 0:
                _save_checkpoint(state, cfg, step)

            step += 1

    _save_checkpoint(state, cfg, step, final=True)
    log.close()
    logger.info("Training complete. Final step: %d", step)


def _save_checkpoint(state, cfg: Config, step: int, final: bool = False) -> None:
    import orbax.checkpoint as ocp
    sub = "final" if final else f"step_{step}"
    ckpt_path = f"{cfg.infra.gcs_bucket.rstrip('/')}/{cfg.infra.checkpoint_dir}/{sub}"
    logger.info("Saving checkpoint → %s", ckpt_path)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(ckpt_path, state.params, force=True)


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to YAML config (e.g. configs/dpo_v5e.yaml).",
)
@click.argument("overrides", nargs=-1)
def main(config: str, overrides: tuple[str, ...]) -> None:
    """Run DPO training on a TPU v5e-8.

    Example::

        tunix-train --config configs/dpo_v5e.yaml training.beta=0.05
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    cfg = load_config(config, list(overrides))
    logger.info("Loaded config from %s", config)
    logger.info("Effective config: %s", asdict(cfg))

    os.makedirs(cfg.infra.log_dir, exist_ok=True)
    run_training(cfg)


if __name__ == "__main__":
    main()
