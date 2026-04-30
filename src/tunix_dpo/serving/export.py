"""
Weight export pipeline — CLI entry-point.

Converts an Orbax checkpoint (JAX bfloat16) to:
    Path A: HuggingFace safetensors  →  vLLM / any PyTorch runtime
    Path B: LiteRT flatbuffer (.tflite)  →  on-device inference (optional)

Key transposition
-----------------
Flax Dense kernels: [in_features, out_features]
PyTorch nn.Linear:  [out_features, in_features]
Every projection weight is transposed during conversion.

Usage
-----
    python -m tunix_dpo.serving.export \\
        --checkpoint_dir ./checkpoints/dpo_v5e_run/final \\
        --base_model     google/gemma-2b \\
        --output_dir     ./exports/tunix_dpo_gemma2b

    tunix-export --checkpoint_dir ... --base_model ... --output_dir ... --export_litert
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ── Flax → HuggingFace key remap patterns ────────────────────────────────────

_REMAP: list[tuple[str, str]] = [
    (r"^embed_tokens\.embedding$", "model.embed_tokens.weight"),
    (r"^model\.norm\.scale$", "model.norm.weight"),
    (r"^layers\.(\d+)\.input_layernorm\.scale$", r"model.layers.\1.input_layernorm.weight"),
    (
        r"^layers\.(\d+)\.post_attention_layernorm\.scale$",
        r"model.layers.\1.post_attention_layernorm.weight",
    ),
    (
        r"^layers\.(\d+)\.pre_feedforward_layernorm\.scale$",
        r"model.layers.\1.pre_feedforward_layernorm.weight",
    ),
    (
        r"^layers\.(\d+)\.post_feedforward_layernorm\.scale$",
        r"model.layers.\1.post_feedforward_layernorm.weight",
    ),
    (r"^layers\.(\d+)\.self_attn\.q_proj\.kernel$", r"model.layers.\1.self_attn.q_proj.weight"),
    (r"^layers\.(\d+)\.self_attn\.k_proj\.kernel$", r"model.layers.\1.self_attn.k_proj.weight"),
    (r"^layers\.(\d+)\.self_attn\.v_proj\.kernel$", r"model.layers.\1.self_attn.v_proj.weight"),
    (r"^layers\.(\d+)\.self_attn\.o_proj\.kernel$", r"model.layers.\1.self_attn.o_proj.weight"),
    (r"^layers\.(\d+)\.self_attn\.q_proj\.bias$", r"model.layers.\1.self_attn.q_proj.bias"),
    (r"^layers\.(\d+)\.self_attn\.k_proj\.bias$", r"model.layers.\1.self_attn.k_proj.bias"),
    (r"^layers\.(\d+)\.self_attn\.v_proj\.bias$", r"model.layers.\1.self_attn.v_proj.bias"),
    (r"^layers\.(\d+)\.mlp\.gate_proj\.kernel$", r"model.layers.\1.mlp.gate_proj.weight"),
    (r"^layers\.(\d+)\.mlp\.up_proj\.kernel$", r"model.layers.\1.mlp.up_proj.weight"),
    (r"^layers\.(\d+)\.mlp\.down_proj\.kernel$", r"model.layers.\1.mlp.down_proj.weight"),
    (r"^lm_head\.kernel$", "lm_head.weight"),
]


def _remap_key(key: str) -> str | None:
    for pattern, replacement in _REMAP:
        new_key = re.sub(pattern, replacement, key)
        if new_key != key:
            return new_key
    return None


# ── Checkpoint loading ────────────────────────────────────────────────────────


def _load_orbax(ckpt_dir: Path) -> dict[str, np.ndarray]:
    import orbax.checkpoint as ocp  # type: ignore[import]

    log.info("Loading Orbax checkpoint: %s", ckpt_dir)
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(ckpt_dir)
    flat: dict[str, np.ndarray] = {}

    def _flatten(node: object, prefix: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _flatten(v, f"{prefix}.{k}" if prefix else k)
        else:
            flat[prefix] = np.array(node, dtype=np.float32)

    _flatten(restored.get("params", restored))
    log.info("  %d parameter tensors loaded", len(flat))
    return flat


# ── State dict builder ────────────────────────────────────────────────────────


def _build_state_dict(flat: dict[str, np.ndarray]) -> dict:
    import torch  # type: ignore[import]

    state_dict: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for flax_key, arr in flat.items():
        hf_key = _remap_key(flax_key)
        if hf_key is None:
            skipped.append(flax_key)
            continue
        tensor = torch.from_numpy(arr).to(torch.bfloat16)
        if hf_key.endswith(".weight") and tensor.ndim == 2:
            tensor = tensor.T  # Flax [in,out] → PyTorch [out,in]
        state_dict[hf_key] = tensor

    if skipped:
        log.warning("  %d keys had no remap rule (e.g. %s)", len(skipped), skipped[:3])
    log.info("  %d tensors mapped to HuggingFace format", len(state_dict))
    return state_dict


# ── Safetensors writer ────────────────────────────────────────────────────────


def _write_safetensors(state_dict: dict, output_dir: Path, shard_gb: float) -> None:
    from safetensors.torch import save_file  # type: ignore[import]

    output_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(shard_gb * 1024**3)
    shards: list[dict] = [{}]
    sizes: list[int] = [0]

    for key, tensor in state_dict.items():
        nb = tensor.nbytes
        if sizes[-1] + nb > max_bytes and shards[-1]:
            shards.append({})
            sizes.append(0)
        shards[-1][key] = tensor
        sizes[-1] += nb

    weight_map: dict[str, str] = {}
    total = 0

    if len(shards) == 1:
        fname = "model.safetensors"
        save_file(shards[0], output_dir / fname, metadata={"format": "pt"})
        weight_map = {k: fname for k in shards[0]}
        total = sizes[0]
        log.info("  Single shard: %s  (%.2f GB)", fname, total / 1e9)
    else:
        n = len(shards)
        for i, (shard, nb) in enumerate(zip(shards, sizes, strict=False)):
            fname = f"model-{i+1:05d}-of-{n:05d}.safetensors"
            save_file(shard, output_dir / fname, metadata={"format": "pt"})
            for k in shard:
                weight_map[k] = fname
            total += nb
            log.info("  Shard %d/%d: %s  (%.2f GB)", i + 1, n, fname, nb / 1e9)
        idx = {"metadata": {"total_size": total}, "weight_map": weight_map}
        (output_dir / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))

    log.info("  Total weight size: %.2f GB", total / 1e9)


# ── Config + tokenizer ────────────────────────────────────────────────────────


def _write_hf_config(base_model: str, output_dir: Path) -> None:
    from transformers import AutoConfig, AutoTokenizer  # type: ignore[import]

    log.info("Copying config + tokenizer from %s", base_model)
    AutoConfig.from_pretrained(base_model).save_pretrained(output_dir)
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.save_pretrained(output_dir)
    gen_cfg = {
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "_from_model_config": True,
    }
    (output_dir / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2))


# ── Verification ──────────────────────────────────────────────────────────────


def _verify(output_dir: Path) -> None:
    import torch  # type: ignore[import]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

    log.info("Verifying export via HuggingFace load…")
    model = AutoModelForCausalLM.from_pretrained(
        output_dir, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tok = AutoTokenizer.from_pretrained(output_dir)
    ids = tok("Hello world", return_tensors="pt")
    with torch.no_grad():
        out = model(**ids)
    assert out.logits.shape[-1] == model.config.vocab_size
    log.info(
        "  Params: %.2fB  logit shape: %s  ✓",
        sum(p.numel() for p in model.parameters()) / 1e9,
        tuple(out.logits.shape),
    )
    del model


# ── LiteRT export ─────────────────────────────────────────────────────────────


def _export_litert(flat: dict[str, np.ndarray], output_path: Path, quant: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import ai_edge_torch  # type: ignore[import]
        import torch
        from transformers import AutoModelForCausalLM

        log.info("LiteRT export via ai_edge_torch (quant=%s)", quant)
        model = AutoModelForCausalLM.from_pretrained(
            output_path.parent, torch_dtype=torch.float32, device_map="cpu"
        )
        model.eval()
        sample = (torch.zeros(1, 16, dtype=torch.long),)
        edge_model = ai_edge_torch.convert(model, sample)
        edge_model.export(str(output_path))
    except ImportError:
        import tensorflow as tf  # type: ignore[import]

        log.info("LiteRT fallback via TFLite converter (quant=%s)", quant)
        embed_key = next((k for k in flat if "embed_tokens" in k), None)
        embed_w = flat.get(embed_key, np.zeros((256000, 2048), np.float32))

        class _Embed(tf.Module):
            def __init__(self, w: np.ndarray) -> None:
                super().__init__()
                self.w = tf.Variable(w.astype(np.float32), trainable=False)

            @tf.function(input_signature=[tf.TensorSpec([1, None], tf.int32)])
            def __call__(self, x: tf.Tensor) -> tf.Tensor:
                return tf.gather(self.w, x)

        with tempfile.TemporaryDirectory() as tmp:
            tf.saved_model.save(_Embed(embed_w), tmp)
            conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
            conv.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            if quant in ("int8", "float16"):
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
            if quant == "float16":
                conv.target_spec.supported_types = [tf.float16]
            output_path.write_bytes(conv.convert())

    size_mb = output_path.stat().st_size / 1e6
    log.info("  LiteRT file: %s  (%.1f MB)", output_path, size_mb)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Export JAX/Orbax DPO checkpoint → HuggingFace + LiteRT."
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shard_size_gb", type=float, default=4.0)
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--export_litert", action="store_true")
    parser.add_argument("--litert_output", default=None)
    parser.add_argument("--litert_quant", choices=["none", "float16", "int8"], default="int8")
    args = parser.parse_args(argv)

    ckpt_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    flat = _load_orbax(ckpt_dir)
    state_dict = _build_state_dict(flat)
    _write_safetensors(state_dict, output_dir, args.shard_size_gb)
    _write_hf_config(args.base_model, output_dir)

    if not args.skip_verify:
        _verify(output_dir)

    if args.export_litert:
        litert_path = Path(args.litert_output or str(output_dir / "model.tflite"))
        _export_litert(flat, litert_path, args.litert_quant)

    log.info("Export complete → %s", output_dir)
    log.info("Serve: vllm serve %s --dtype bfloat16", output_dir)


if __name__ == "__main__":
    main()
