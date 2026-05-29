"""
DPO training CLI — registered as ``tunix-train``.

Trains ``google/gemma-3-1b-it`` on TPU v5e using PyTorch + torch/xla.

WHY PYTORCH AND NOT JAX/FLAX
----------------------------
Gemma 3 has no Flax implementation in HuggingFace transformers as of v4.50+.
The supported path for Gemma 3 on TPU is PyTorch with torch/xla, which
compiles operations to the same XLA backend that JAX uses — so we get the
same TPU performance with a model architecture that actually exists.

The pure DPO loss functions in ``losses.py`` are still written in JAX for
testability (they're algorithm, not infrastructure). The training step
re-implements the same math in torch so it can run inside the XLA graph.
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


# ── Config loading ────────────────────────────────────────────────────────────

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


# ── Loss helpers (pure torch, mirrors training/losses.py JAX implementation) ──

def _log_probs_from_logits_torch(logits, labels, mask):
    """Sum log p(labels | context) over response tokens. Pure torch.

    Shifts: logits[t-1] predicts labels[t]. Drops the last logit and the
    first label/mask position.
    """
    import torch
    import torch.nn.functional as F

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    shifted_mask = mask[:, 1:].to(shifted_logits.dtype)

    log_probs = F.log_softmax(shifted_logits, dim=-1)
    selected = torch.gather(log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
    return (selected * shifted_mask).sum(dim=-1)


def _dpo_step(model, ref_model, batch, beta: float):
    """One forward+backward DPO step in torch. Returns (loss, metrics_dict)."""
    import torch
    import torch.nn.functional as F

    chosen_ids = batch["chosen_ids"]
    rejected_ids = batch["rejected_ids"]
    chosen_mask = batch["chosen_mask"]
    rejected_mask = batch["rejected_mask"]

    # Policy forward — gradients flow
    pol_chosen_logits = model(chosen_ids).logits
    pol_rejected_logits = model(rejected_ids).logits

    # Reference forward — no gradients
    with torch.no_grad():
        ref_chosen_logits = ref_model(chosen_ids).logits
        ref_rejected_logits = ref_model(rejected_ids).logits

    pol_chosen = _log_probs_from_logits_torch(pol_chosen_logits, chosen_ids, chosen_mask)
    pol_rejected = _log_probs_from_logits_torch(pol_rejected_logits, rejected_ids, rejected_mask)
    ref_chosen = _log_probs_from_logits_torch(ref_chosen_logits, chosen_ids, chosen_mask)
    ref_rejected = _log_probs_from_logits_torch(ref_rejected_logits, rejected_ids, rejected_mask)

    chosen_logratio = pol_chosen - ref_chosen
    rejected_logratio = pol_rejected - ref_rejected

    logits = beta * (chosen_logratio - rejected_logratio)
    loss = -F.logsigmoid(logits).mean()

    metrics = {
        "reward_margin": (chosen_logratio - rejected_logratio).mean().detach(),
        "reward_acc": (chosen_logratio > rejected_logratio).float().mean().detach(),
        "policy_chosen_logp": pol_chosen.mean().detach(),
        "policy_rejected_logp": pol_rejected.mean().detach(),
    }
    return loss, metrics


# ── Training loop ─────────────────────────────────────────────────────────────

def run_training(cfg: Config) -> None:
    """Train Gemma 3 1B IT with DPO on TPU v5e via torch/xla."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tunix_dpo.training.dataset import DataLoader, HHRLHFDataset
    from tunix_dpo.training.logger import TrajectoryLogger
    from tunix_dpo.training.optimizer import make_optimizer_torch

    # Detect TPU vs CPU/GPU fallback
    device = _detect_device()
    n_devices = _world_size()
    logger.info("Device: %s | world_size: %d", device, n_devices)

    # ── Tokenizer + base model + frozen reference copy ───────────────────────
    logger.info("Loading tokenizer: %s", cfg.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _dtype(cfg.model.dtype)
    logger.info("Loading model in %s …", cfg.model.dtype)

    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name, torch_dtype=dtype)
    model.to(device)
    model.train()

    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name, torch_dtype=dtype)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_path = Path(cfg.data.data_dir) / cfg.data.train_file
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            f"Did you run `tunix-curate --output_dir {cfg.data.data_dir}` first?"
        )

    train_ds = HHRLHFDataset(train_path, tokenizer, cfg.model.max_seq_len)
    global_bs = cfg.training.per_device_batch_size * n_devices

    if len(train_ds) == 0:
        raise RuntimeError(
            f"Dataset at {train_path} is empty. "
            f"Check that curation produced non-zero records."
        )
    if len(train_ds) < global_bs:
        raise RuntimeError(
            f"Dataset has {len(train_ds)} examples but global batch size is "
            f"{global_bs}. With drop_last=True the loader will yield zero "
            f"batches. Either curate more data or reduce per_device_batch_size."
        )

    loader = DataLoader(train_ds, batch_size=global_bs, seed=cfg.infra.seed)

    total_steps = len(loader) * cfg.training.num_epochs
    logger.info(
        "Dataset: %d examples, %d batches/epoch, %d total steps (global_batch=%d)",
        len(train_ds), len(loader), total_steps, global_bs,
    )
    optimizer, scheduler = make_optimizer_torch(model, cfg.training, total_steps)

    # ── Trajectory logging ───────────────────────────────────────────────────
    log = TrajectoryLogger(cfg.infra.log_dir)
    log.log(0, {"event": "start", "total_steps": total_steps, "global_batch": global_bs})

    step = 0
    for epoch in range(cfg.training.num_epochs):
        logger.info("── Epoch %d/%d ──", epoch + 1, cfg.training.num_epochs)
        for batch_np in loader:
            if step == 0:
                logger.info("First batch loaded; starting forward pass. "
                            "On TPU the first step compiles XLA — expect "
                            "2–5 min of silence before the first log line.")

            # Convert batch to torch tensors on device.
            # Token-ID arrays MUST be int64 — torch's embedding backward fails
            # with int32. Mask arrays stay int32 (just multiplied with logits).
            batch = {}
            for k, v in batch_np.items():
                if k.endswith("_ids"):
                    batch[k] = torch.as_tensor(v, dtype=torch.long, device=device)
                else:
                    batch[k] = torch.as_tensor(v, device=device)

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _dpo_step(model, ref_model, batch, beta=cfg.training.beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            optimizer.step()
            scheduler.step()
            _xla_mark_step()

            if step == 0:
                logger.info(
                    "First step complete: loss=%.4f reward_acc=%.3f",
                    float(loss.detach().cpu()),
                    float(metrics["reward_acc"].cpu()),
                )

            if step % cfg.infra.log_every_steps == 0:
                row = {"dpo_loss": float(loss.detach().cpu())}
                row.update({k: float(v.cpu()) for k, v in metrics.items()})
                row["lr"] = scheduler.get_last_lr()[0]
                log.log(step, row)

            if step > 0 and step % cfg.infra.checkpoint_every_steps == 0:
                _save_checkpoint(model, tokenizer, cfg, step)

            step += 1

    _save_checkpoint(model, tokenizer, cfg, step, final=True)
    log.close()
    logger.info("Training complete. Final step: %d", step)


def _save_checkpoint(model, tokenizer, cfg: Config, step: int, final: bool = False) -> None:
    """Save HuggingFace-format checkpoint to GCS (or local path)."""
    sub = "final" if final else f"step_{step}"
    ckpt_path = f"{cfg.infra.gcs_bucket.rstrip('/')}/{cfg.infra.checkpoint_dir}/{sub}"
    logger.info("Saving checkpoint → %s", ckpt_path)

    # save_pretrained handles gs:// paths via huggingface_hub when available;
    # fall back to local + manual upload if not.
    try:
        model.save_pretrained(ckpt_path, safe_serialization=True)
        tokenizer.save_pretrained(ckpt_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("Direct GCS save failed (%s); writing locally then uploading", e)
        local = Path(f"./checkpoints/{sub}")
        local.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(local, safe_serialization=True)
        tokenizer.save_pretrained(local)
        _gsutil_upload(local, ckpt_path)


def _gsutil_upload(local: Path, gcs_path: str) -> None:
    import subprocess
    if not gcs_path.startswith("gs://"):
        return
    subprocess.run(["gsutil", "-m", "cp", "-r", str(local), gcs_path], check=False)


# ── Device / environment helpers ──────────────────────────────────────────────
#
# torch_xla deprecated several xm.* symbols in 2.5 (and removed xrt_world_size
# entirely). These helpers try the new names first, then fall back to the
# legacy ones — so the same code works against any 2.4+ release.

def _detect_device():
    """Return the right torch device: xla on TPU, cuda on GPU, cpu otherwise."""
    try:
        import torch_xla  # type: ignore[import-not-found]
        # torch_xla >= 2.5 exposes torch_xla.device() directly
        if hasattr(torch_xla, "device"):
            return torch_xla.device()
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
        return xm.xla_device()
    except ImportError:
        pass
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _world_size() -> int:
    """Number of devices (TPU chips, GPUs, or 1 on CPU).

    torch_xla 2.5+ moved this to ``torch_xla.runtime.world_size``;
    older releases had it on ``xm.xrt_world_size``.
    """
    try:
        import torch_xla.runtime as xr  # type: ignore[import-not-found]
        if hasattr(xr, "world_size"):
            return int(xr.world_size())
    except ImportError:
        pass
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
        if hasattr(xm, "xrt_world_size"):
            return int(xm.xrt_world_size())
    except ImportError:
        pass
    import torch
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def _xla_mark_step() -> None:
    """Tell torch/xla to finalize the current step's graph and execute it.

    ``torch_xla.sync()`` is the modern name (2.5+); ``xm.mark_step()`` still
    works on both old and new releases, so we prefer it for compatibility.
    """
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
        xm.mark_step()
        return
    except ImportError:
        pass
    try:
        import torch_xla  # type: ignore[import-not-found]
        if hasattr(torch_xla, "sync"):
            torch_xla.sync()
    except ImportError:
        pass


def _dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


# ── CLI entry point ──────────────────────────────────────────────────────────

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
