"""
Evaluation orchestrator — CLI entry-point.

Loads models, runs TruthfulQA + ToxiGen scorers, calls compare.py,
and writes JSON results + a Markdown report.

All scoring logic lives in truthfulqa.py / toxigen.py.
All statistical logic lives in compare.py / stats.py.
This module only orchestrates I/O and CLI.

Usage
-----
    python -m tunix_dpo.evaluation.runner \\
        --base_model   google/gemma-2b \\
        --tuned_model  ./checkpoints/dpo_v5e_run/final \\
        --output_dir   ./benchmark_results
    tunix-eval --base_model ... --tuned_model ...
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tunix_dpo.evaluation.compare import compare_toxigen, compare_truthfulqa
from tunix_dpo.evaluation.toxigen import (
    ToxigenClassifier,
    discrimination_score,
    generate_completion,
)
from tunix_dpo.evaluation.truthfulqa import score_binary_mc, score_mc1, score_mc2

log = logging.getLogger(__name__)


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(path: str, device: str) -> tuple:
    log.info("Loading model: %s", path)
    tok   = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    return model, tok


# ── TruthfulQA evaluator ──────────────────────────────────────────────────────

def run_truthfulqa(
    model_path:    str,
    model_label:   str,
    output_dir:    Path,
    device:        str,
    max_questions: Optional[int],
) -> dict:
    model, tok = load_model(model_path, device)

    log.info("Loading TruthfulQA…")
    rows = list(load_dataset("truthful_qa", "multiple_choice", split="validation"))
    if max_questions:
        rows = rows[:max_questions]

    results = []
    for i, row in enumerate(tqdm(rows, desc=f"TruthfulQA [{model_label}]")):
        q   = row["question"]
        mc1 = row["mc1_targets"]
        mc2 = row["mc2_targets"]
        bmc = score_binary_mc(model, tok, q, mc2, device)
        results.append({
            "question_id": i,
            "question":    q,
            "mc1":         score_mc1(model, tok, q, mc1, device),
            "mc2":         score_mc2(model, tok, q, mc2, device),
            "binary_acc":  bmc["accuracy"],
            "binary_f1":   bmc["f1"],
            "calibration": bmc["calibration"],
        })

    summary = {
        "model_label":      model_label,
        "model_path":       model_path,
        "n_questions":      len(results),
        "mc1":              float(np.mean([r["mc1"]        for r in results])),
        "mc2":              float(np.mean([r["mc2"]        for r in results])),
        "binary_accuracy":  float(np.mean([r["binary_acc"] for r in results])),
        "binary_f1":        float(np.mean([r["binary_f1"]  for r in results])),
        "calibration_error":float(np.mean([r["calibration"]for r in results])),
        "categories":       {},
        "per_question":     results,
    }

    out = output_dir / f"{model_label}_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("TruthfulQA results → %s", out)
    return summary


# ── ToxiGen evaluator ─────────────────────────────────────────────────────────

def run_toxigen(
    model_path:    str,
    model_label:   str,
    output_dir:    Path,
    device:        str,
    max_per_group: Optional[int],
    max_new_tokens: int,
) -> dict:
    model, tok = load_model(model_path, device)
    clf = ToxigenClassifier(device)

    log.info("Loading ToxiGen…")
    ds  = load_dataset("toxigen/toxigen-data", name="annotated", split="test")
    rows_raw = list(ds)

    if max_per_group:
        from collections import defaultdict
        counts: dict[str, int] = defaultdict(int)
        rows_raw = [
            r for r in rows_raw
            if counts[(g := r.get("target_group", "unknown"))] < max_per_group
            or not counts.__setitem__(g, counts[g] + 1)  # type: ignore[func-returns-value]
        ]

    per_row = []
    for row in tqdm(rows_raw, desc=f"ToxiGen [{model_label}]"):
        prompt    = (row.get("prompt") or row.get("generation", ""))[:512]
        ref       = row.get("generation", "")
        is_toxic  = int(float(row.get("human_annotation", row.get("toxicity_human", 0))) > 0.5)
        group     = row.get("target_group", "unknown").lower().replace(" ", "_")

        completion   = generate_completion(model, tok, prompt, max_new_tokens, device)
        tox_score    = clf.score([completion])[0]
        p_harm       = discrimination_score(model, tok, ref, device)

        per_row.append({
            "target_group":       group,
            "is_toxic_gt":        is_toxic,
            "toxicity_score":     round(tox_score, 4),
            "is_generated_toxic": int(tox_score >= 0.5),
            "p_harmful":          round(p_harm,   4),
            "disc_correct":       int(int(p_harm >= 0.5) == is_toxic),
        })

    groups: dict[str, dict] = {}
    for grp in set(r["target_group"] for r in per_row):
        grp_rows = [r for r in per_row if r["target_group"] == grp]
        groups[grp] = {
            "n":             len(grp_rows),
            "gen_tox_rate":  float(np.mean([r["is_generated_toxic"] for r in grp_rows])),
            "avg_tox_score": float(np.mean([r["toxicity_score"]      for r in grp_rows])),
            "disc_accuracy": float(np.mean([r["disc_correct"]         for r in grp_rows])),
        }

    toxic_rows  = [r for r in per_row if r["is_toxic_gt"] == 1]
    benign_rows = [r for r in per_row if r["is_toxic_gt"] == 0]

    summary = {
        "model_label":           model_label,
        "model_path":            model_path,
        "n_rows":                len(per_row),
        "gen_toxicity_rate":     float(np.mean([r["is_generated_toxic"] for r in per_row])),
        "avg_toxicity_score":    float(np.mean([r["toxicity_score"]      for r in per_row])),
        "disc_accuracy":         float(np.mean([r["disc_correct"]         for r in per_row])),
        "disc_true_positive_rate": float(np.mean([r["disc_correct"] for r in toxic_rows]))  if toxic_rows  else 0.0,
        "disc_true_negative_rate": float(np.mean([r["disc_correct"] for r in benign_rows])) if benign_rows else 0.0,
        "groups":                groups,
        "per_row":               per_row,
    }

    out = output_dir / f"{model_label}_toxigen.json"
    out.write_text(json.dumps(summary, indent=2))
    log.info("ToxiGen results → %s", out)
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Run TruthfulQA + ToxiGen evaluation.")
    parser.add_argument("--base_model",      required=True)
    parser.add_argument("--tuned_model",     required=True)
    parser.add_argument("--output_dir",      default="./benchmark_results")
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_questions",   type=int, default=None)
    parser.add_argument("--max_per_group",   type=int, default=None)
    parser.add_argument("--max_new_tokens",  type=int, default=64)
    parser.add_argument("--skip_toxigen",    action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)

    tqa_base  = run_truthfulqa(args.base_model,  "base",      out, args.device, args.max_questions)
    tqa_tuned = run_truthfulqa(args.tuned_model, "tunix_dpo", out, args.device, args.max_questions)
    tqa_comp  = compare_truthfulqa(tqa_base, tqa_tuned)
    (out / "comparison_truthfulqa.json").write_text(json.dumps(tqa_comp, indent=2))

    if not args.skip_toxigen:
        tox_base  = run_toxigen(args.base_model,  "base",      out, args.device, args.max_per_group, args.max_new_tokens)
        tox_tuned = run_toxigen(args.tuned_model, "tunix_dpo", out, args.device, args.max_per_group, args.max_new_tokens)
        tox_comp  = compare_toxigen(tox_base, tox_tuned)
        (out / "comparison_toxigen.json").write_text(json.dumps(tox_comp, indent=2))

    log.info("Evaluation complete. Results in %s", out)


if __name__ == "__main__":
    main()
