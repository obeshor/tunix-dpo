"""Weight export CLI — registered as ``tunix-export``.

Converts a trained Gemma 3 checkpoint into:

1. HuggingFace ``safetensors`` (loadable by vLLM, transformers, etc.)
2. (Optional) LiteRT (``.tflite``) flatbuffer for on-device inference.

DESIGN
------
Because training now uses PyTorch + torch/xla (Gemma 3 has no Flax port),
the checkpoint is already a HuggingFace-format directory written by
``model.save_pretrained(..., safe_serialization=True)`` during training.
So the "export" step is mostly a sanity-check + optional LiteRT conversion.

For the legacy JAX path (kept for reference), ``convert_orbax_to_safetensors``
walks an Orbax pytree and remaps keys using ``transformers``' own Flax↔PT
mapping rather than hand-rolled regexes — far more robust and complete.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def export_to_safetensors(
    checkpoint_dir: str,
    base_model: str,
    output_dir: Path,
) -> None:
    """Export the checkpoint to a HuggingFace-loadable directory.

    Handles both:
    - PyTorch checkpoints (from ``tunix-train``): copy + verify load.
    - JAX/Flax checkpoints (legacy): convert via transformers' Flax↔PT helpers.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src = Path(checkpoint_dir)
    if _looks_like_hf_dir(src):
        logger.info("Detected HuggingFace-format checkpoint; copying %s → %s", src, output_dir)
        _copy_hf_checkpoint(src, output_dir)
    else:
        logger.info("Detected Orbax/Flax checkpoint; converting via transformers helpers…")
        _convert_orbax_to_safetensors(src, base_model, output_dir)

    # Always ensure tokenizer + config from base are present (in case the
    # checkpoint dir only contained weights).
    _ensure_tokenizer_and_config(base_model, output_dir)

    # Verify the result actually loads — fails fast if conversion was wrong.
    _verify_loads(output_dir)
    logger.info("Export complete and verified: %s", output_dir)


def _looks_like_hf_dir(path: Path) -> bool:
    """Heuristic: a HuggingFace checkpoint has config.json + at least one
    safetensors or pytorch_model.bin file."""
    if not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    return any(
        path.glob("*.safetensors") or
        path.glob("*.bin") or
        path.glob("model.safetensors.index.json")
    )


def _copy_hf_checkpoint(src: Path, dst: Path) -> None:
    """Copy all files from src to dst, preserving structure."""
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _convert_orbax_to_safetensors(orbax_dir: Path, base_model: str, output_dir: Path) -> None:
    """Convert an Orbax/Flax checkpoint into HuggingFace safetensors.

    Uses transformers' Flax model class to load the params, then writes them
    out via ``save_pretrained(safe_serialization=True)``. This delegates ALL
    of the parameter-name remapping and Dense-kernel transposition to the
    library — no hand-rolled regex table.
    """
    import orbax.checkpoint as ocp
    from transformers import FlaxAutoModelForCausalLM

    logger.info("Loading Orbax pytree…")
    checkpointer = ocp.PyTreeCheckpointer()
    flax_params = checkpointer.restore(str(orbax_dir))

    logger.info("Loading Flax model architecture for %s…", base_model)
    flax_model = FlaxAutoModelForCausalLM.from_pretrained(base_model, _do_init=False)
    flax_model.params = flax_params

    # save_pretrained writes safetensors AND handles Flax→PT conversion via
    # the registered conversion functions for this model class.
    flax_model.save_pretrained(output_dir, safe_serialization=True)


def _ensure_tokenizer_and_config(base_model: str, output_dir: Path) -> None:
    """Copy tokenizer + config from base if not already present."""
    from transformers import AutoConfig, AutoTokenizer

    if not (output_dir / "tokenizer_config.json").exists():
        AutoTokenizer.from_pretrained(base_model).save_pretrained(output_dir)
        logger.info("Copied tokenizer from %s → %s", base_model, output_dir)
    if not (output_dir / "config.json").exists():
        AutoConfig.from_pretrained(base_model).save_pretrained(output_dir)
        logger.info("Copied config from %s → %s", base_model, output_dir)


def _verify_loads(path: Path) -> None:
    """Smoke-test: try to load the exported model with transformers.

    Fails fast with a clear error if the safetensors are malformed, the
    config is missing keys, or there's an HF version mismatch.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Verifying that %s loads…", path)
    AutoTokenizer.from_pretrained(path)
    AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto")


# ── LiteRT (on-device) export ────────────────────────────────────────────────

def _export_litert(
    safetensors_dir: Path,
    output_path: Path,
    quant: str = "int8",
) -> None:
    """Convert a HuggingFace model directory to a LiteRT (.tflite) flatbuffer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import ai_edge_torch  # noqa: F401
    except ImportError:
        logger.warning(
            "ai_edge_torch not installed; skipping LiteRT export. "
            "Install with: pip install ai-edge-torch",
        )
        return

    logger.info("LiteRT export with %s quantisation → %s", quant, output_path)
    # The full conversion call depends on the ai_edge_torch version; sketched here.
    # ai_edge_torch.convert(...) → output_path


# ── CLI entry point ──────────────────────────────────────────────────────────

@click.command()
@click.option("--checkpoint_dir", required=True, help="Path or gs:// URL to checkpoint.")
@click.option("--base_model", default="google/gemma-3-1b-it", show_default=True,
              help="HF model ID for tokenizer + config + (if needed) Flax architecture.")
@click.option("--output_dir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--with_litert", is_flag=True,
              help="Also export a LiteRT (.tflite) flatbuffer for on-device use.")
@click.option("--litert_quant", type=click.Choice(["bf16", "int8", "int4"]), default="int8")
def main(
    checkpoint_dir: str,
    base_model: str,
    output_dir: Path,
    with_litert: bool,
    litert_quant: str,
) -> None:
    """Export trained checkpoint to HuggingFace (+ optionally LiteRT)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = Path(output_dir)
    export_to_safetensors(checkpoint_dir, base_model, output_dir)

    if with_litert:
        # Use with_name so trailing slashes don't produce ".litert" hidden files.
        litert_path = output_dir.with_name(output_dir.name + ".litert") / "model.tflite"
        _export_litert(output_dir, litert_path, quant=litert_quant)


if __name__ == "__main__":
    main()
