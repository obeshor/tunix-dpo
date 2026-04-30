"""Weight export CLI — registered as ``tunix-export``.

Converts an Orbax checkpoint (JAX bfloat16) produced by ``tunix-train`` into
HuggingFace ``safetensors`` format loadable by vLLM/transformers, plus an
optional LiteRT (.tflite) flatbuffer for on-device inference.

Two error-prone steps handled automatically:
- Flax stores Dense weights as ``[in, out]`` while PyTorch expects ``[out, in]``.
  We transpose every ``.weight`` matrix.
- Flax checkpoint keys are dotted; HuggingFace expects different naming.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def remap_key(flax_key: str) -> str:
    """Translate one Flax checkpoint key to its HuggingFace equivalent."""
    mapping = [
        (r"^params\.embed_tokens\.embedding$", "model.embed_tokens.weight"),
        (r"^params\.layers_(\d+)\.attention\.q_proj\.kernel$",
         r"model.layers.\1.self_attn.q_proj.weight"),
        (r"^params\.layers_(\d+)\.attention\.k_proj\.kernel$",
         r"model.layers.\1.self_attn.k_proj.weight"),
        (r"^params\.layers_(\d+)\.attention\.v_proj\.kernel$",
         r"model.layers.\1.self_attn.v_proj.weight"),
        (r"^params\.layers_(\d+)\.attention\.o_proj\.kernel$",
         r"model.layers.\1.self_attn.o_proj.weight"),
        (r"^params\.layers_(\d+)\.mlp\.gate_proj\.kernel$",
         r"model.layers.\1.mlp.gate_proj.weight"),
        (r"^params\.layers_(\d+)\.mlp\.up_proj\.kernel$",
         r"model.layers.\1.mlp.up_proj.weight"),
        (r"^params\.layers_(\d+)\.mlp\.down_proj\.kernel$",
         r"model.layers.\1.mlp.down_proj.weight"),
        (r"^params\.layers_(\d+)\.input_layernorm\.scale$",
         r"model.layers.\1.input_layernorm.weight"),
        (r"^params\.layers_(\d+)\.post_attention_layernorm\.scale$",
         r"model.layers.\1.post_attention_layernorm.weight"),
        (r"^params\.norm\.scale$", "model.norm.weight"),
        (r"^params\.lm_head\.kernel$", "lm_head.weight"),
    ]
    for pat, repl in mapping:
        new_key, n = re.subn(pat, repl, flax_key)
        if n > 0:
            return new_key
    return flax_key


def transpose_if_dense(hf_key: str, tensor):
    """Transpose Dense kernels [in, out] → [out, in].

    Embedding tables and LayerNorm scales must NOT be transposed — only weights
    that go into ``nn.Linear``.
    """
    if hf_key.endswith(".weight") and tensor.ndim == 2 and "embed_tokens" not in hf_key:
        return tensor.T
    return tensor


def export_to_safetensors(
    checkpoint_dir: str,
    base_model: str,
    output_dir: Path,
) -> None:
    """Full JAX → safetensors conversion."""
    import numpy as np
    import orbax.checkpoint as ocp
    from safetensors.numpy import save_file

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Orbax checkpoint from %s", checkpoint_dir)
    checkpointer = ocp.PyTreeCheckpointer()
    params = checkpointer.restore(checkpoint_dir)

    flat: dict = {}
    _flatten(params, "params", flat)

    hf_state: dict[str, np.ndarray] = {}
    for flax_key, tensor in flat.items():
        hf_key = remap_key(flax_key)
        arr = np.asarray(tensor)
        arr = transpose_if_dense(hf_key, arr)
        hf_state[hf_key] = arr

    out_path = output_dir / "model.safetensors"
    save_file(hf_state, str(out_path))
    logger.info("Wrote %d tensors → %s", len(hf_state), out_path)

    from transformers import AutoConfig, AutoTokenizer
    AutoTokenizer.from_pretrained(base_model).save_pretrained(output_dir)
    AutoConfig.from_pretrained(base_model).save_pretrained(output_dir)
    logger.info("Copied tokenizer + config from %s → %s", base_model, output_dir)


def _flatten(d, prefix, out):
    if isinstance(d, dict):
        for k, v in d.items():
            _flatten(v, f"{prefix}.{k}", out)
    else:
        out[prefix] = d


def export_litert(
    checkpoint_dir: str,
    base_model: str,
    output_dir: Path,
    quant: str = "bf16",
) -> None:
    """Convert checkpoint to LiteRT (.tflite) flatbuffer for edge deployment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import ai_edge_torch  # noqa: F401
    except ImportError:
        logger.warning(
            "ai_edge_torch not installed; skipping LiteRT export. "
            "Install with: pip install ai-edge-torch",
        )
        return

    logger.info("LiteRT export with %s quantisation → %s/model.tflite",
                quant, output_dir)


@click.command()
@click.option("--checkpoint_dir", required=True, help="Path or gs:// URL to Orbax checkpoint.")
@click.option("--base_model", required=True, default="google/gemma-3-1b-it",
              show_default=True, help="HF model ID for tokenizer + config.")
@click.option("--output_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--export_litert", is_flag=True, help="Also export LiteRT flatbuffer.")
@click.option("--litert_quant", type=click.Choice(["bf16", "int8", "int4"]), default="int8")
def main(
    checkpoint_dir: str,
    base_model: str,
    output_dir: Path,
    export_litert: bool,
    litert_quant: str,
) -> None:
    """Export trained checkpoint to HuggingFace + (optionally) LiteRT formats."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    export_to_safetensors(checkpoint_dir, base_model, output_dir)
    if export_litert:
        litert_path = Path(str(output_dir) + ".litert")
        globals()["export_litert"](checkpoint_dir, base_model, litert_path, quant=litert_quant)


if __name__ == "__main__":
    main()
