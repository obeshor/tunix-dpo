"""
DPO training loop — CLI entry-point.

All heavy logic lives in the sibling modules:
    config.py   — dataclasses
    losses.py   — pure JAX loss functions
    dataset.py  — HHRLHFDataset + DataLoader
    optimizer.py — make_optimizer()
    logger.py   — TrajectoryLogger

This module orchestrates them and adds pmap, checkpointing, and eval.

Usage
-----
    python -m tunix_dpo.training.train --config configs/dpo_v5e.yaml
    tunix-train --config configs/dpo_v5e.yaml training.beta=0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from transformers import AutoTokenizer

from tunix_dpo.training.config import Config
from tunix_dpo.training.dataset import HHRLHFDataset, make_dataloader
from tunix_dpo.training.logger import TrajectoryLogger
from tunix_dpo.training.losses import dpo_loss, log_probs_from_logits, sft_loss
from tunix_dpo.training.optimizer import make_optimizer

log = logging.getLogger(__name__)


# ── pmap training step ────────────────────────────────────────────────────────

def make_train_step(
    beta: float,
    sft_weight: float,
    log_sft: bool,
) -> ...:  # type: ignore[type-arg]
    """Return a pmap-compiled training step closure."""

    @jax.jit
    def _step(state: train_state.TrainState, ref_params: dict, batch: dict) -> tuple:
        def loss_fn(params: dict) -> tuple[jnp.ndarray, dict]:
            chosen_logits   = state.apply_fn({"params": params}, batch["chosen_input_ids"]).logits
            rejected_logits = state.apply_fn({"params": params}, batch["rejected_input_ids"]).logits

            policy_chosen_lp   = log_probs_from_logits(chosen_logits,   batch["chosen_labels"])
            policy_rejected_lp = log_probs_from_logits(rejected_logits, batch["rejected_labels"])

            ref_chosen_lp   = jax.lax.stop_gradient(
                log_probs_from_logits(
                    state.apply_fn({"params": ref_params}, batch["chosen_input_ids"]).logits,
                    batch["chosen_labels"],
                )
            )
            ref_rejected_lp = jax.lax.stop_gradient(
                log_probs_from_logits(
                    state.apply_fn({"params": ref_params}, batch["rejected_input_ids"]).logits,
                    batch["rejected_labels"],
                )
            )

            d_loss, chosen_r, rejected_r = dpo_loss(
                policy_chosen_lp, policy_rejected_lp,
                ref_chosen_lp,    ref_rejected_lp,
                beta=beta,
            )

            s_loss = (
                sft_loss(chosen_logits, batch["chosen_labels"])
                if log_sft else jnp.zeros(())
            )

            total = d_loss + sft_weight * s_loss
            metrics = {
                "dpo_loss":        d_loss,
                "sft_loss":        s_loss,
                "total_loss":      total,
                "chosen_reward":   chosen_r.mean(),
                "rejected_reward": rejected_r.mean(),
                "reward_margin":   (chosen_r - rejected_r).mean(),
                "reward_accuracy": (chosen_r > rejected_r).astype(jnp.float32).mean(),
            }
            return total, metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads   = jax.lax.pmean(grads,   axis_name="batch")
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return state.apply_gradients(grads=grads), metrics

    return jax.pmap(_step, axis_name="batch")


# ── Main training function ────────────────────────────────────────────────────

def train(cfg: Config) -> None:
    devices   = jax.devices()
    n_devices = len(devices)
    log.info("JAX devices: %d  (%s)", n_devices, devices[0].device_kind)

    global_batch = cfg.training.per_device_batch_size * n_devices

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = HHRLHFDataset(
        Path(cfg.data.data_dir) / cfg.data.train_file,
        tokenizer, cfg.model.max_seq_len,
    )
    eval_ds = HHRLHFDataset(
        Path(cfg.data.data_dir) / cfg.data.eval_file,
        tokenizer, cfg.model.max_seq_len,
    )

    train_loader = make_dataloader(train_ds, global_batch, shuffle=True,  num_workers=cfg.data.num_workers)
    eval_loader  = make_dataloader(eval_ds,  global_batch, shuffle=False, num_workers=cfg.data.num_workers)

    steps_per_epoch = len(train_loader) // cfg.training.gradient_accumulation_steps
    total_steps     = steps_per_epoch * cfg.training.num_epochs

    try:
        import tunix  # type: ignore[import]
        model  = tunix.load_model(cfg.model.model_name, dtype=cfg.model.dtype)
        params = model.init_params(jax.random.PRNGKey(cfg.infra.seed))
    except ImportError:
        from transformers import FlaxAutoModelForCausalLM  # type: ignore[import]
        hf_model = FlaxAutoModelForCausalLM.from_pretrained(cfg.model.model_name, dtype=jnp.bfloat16)
        model, params = hf_model.module, hf_model.params

    ref_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
    optimizer  = make_optimizer(cfg.training, total_steps)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    state      = jax.device_put_replicated(state,      devices)
    ref_params = jax.device_put_replicated(ref_params, devices)

    p_train_step = make_train_step(
        cfg.training.beta, cfg.training.sft_loss_weight, cfg.training.log_sft_loss
    )
    checkpointer = ocp.StandardCheckpointer()
    ckpt_dir     = Path(cfg.infra.checkpoint_dir)

    with TrajectoryLogger(cfg.infra.log_dir) as tlog:
        global_step = 0
        for epoch in range(cfg.training.num_epochs):
            log.info("── Epoch %d/%d ──", epoch + 1, cfg.training.num_epochs)

            for batch in train_loader:
                batch = {
                    k: v.reshape((n_devices, cfg.training.per_device_batch_size, -1))
                    for k, v in batch.items()
                }
                state, metrics = p_train_step(state, ref_params, batch)
                global_step += 1

                if global_step % cfg.infra.log_every_steps == 0:
                    m = {k: float(v[0]) for k, v in metrics.items()}
                    tlog.log(global_step, m, prefix="train")
                    log.info(
                        "step=%d  dpo=%.4f  sft=%.4f  reward_acc=%.3f",
                        global_step, m["dpo_loss"], m["sft_loss"], m["reward_accuracy"],
                    )

                if global_step % cfg.infra.eval_every_steps == 0:
                    eval_m = _eval(state, ref_params, eval_loader, p_train_step, n_devices, cfg)
                    tlog.log(global_step, eval_m, prefix="eval")

                if global_step % cfg.infra.checkpoint_every_steps == 0:
                    ckpt = ckpt_dir / f"step_{global_step:06d}"
                    checkpointer.save(
                        ckpt,
                        jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state)),
                    )

        checkpointer.save(
            ckpt_dir / "final",
            jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state)),
        )

    log.info("Training complete.")


def _eval(state, ref_params, loader, step_fn, n_devices, cfg: Config) -> dict:
    accum: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = {
            k: v.reshape((n_devices, cfg.training.per_device_batch_size, -1))
            for k, v in batch.items()
        }
        _, metrics = step_fn(state, ref_params, batch)
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + float(v[0])
        n += 1
    return {k: v / n for k, v in accum.items()}


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str, overrides: list[str]) -> Config:
    import yaml  # type: ignore[import]

    cfg = Config()
    if Path(path).exists():
        with open(path) as f:
            raw = yaml.safe_load(f)
        for section, values in (raw or {}).items():
            sub = getattr(cfg, section, None)
            if sub and isinstance(values, dict):
                for k, v in values.items():
                    setattr(sub, k, v)

    for ov in overrides:
        if "=" in ov:
            dotted, val = ov.lstrip("-").split("=", 1)
            section, attr = dotted.split(".", 1)
            sub = getattr(cfg, section, None)
            if sub:
                setattr(sub, attr, type(getattr(sub, attr))(val))
    return cfg


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="DPO training on TPU via Tunix/JAX.")
    parser.add_argument("--config", default="configs/dpo_v5e.yaml")
    args, overrides = parser.parse_known_args(argv)

    cfg = load_config(args.config, overrides)
    log.info("Config:\n%s", json.dumps(asdict(cfg), indent=2))
    train(cfg)


if __name__ == "__main__":
    main()
