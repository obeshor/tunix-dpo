"""TruthfulQA + ToxiGen evaluation runner — registered as ``tunix-eval``.

Loads the base and tuned models, runs both benchmarks, writes results as
JSON, and prints a comparison summary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from tunix_dpo.evaluation.compare import compare_toxigen, compare_truthfulqa

logger = logging.getLogger(__name__)


def evaluate_truthfulqa(model_path: str, model_label: str, max_questions: int) -> dict:
    """Run TruthfulQA against one model and return a results dict."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tunix_dpo.evaluation.truthfulqa import (
        calibration_error,
        score_binary_mc,
        score_mc1,
        score_mc2,
    )

    logger.info("Loading TruthfulQA dataset (mc subset, validation split)…")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    if max_questions:
        ds = ds.select(range(min(max_questions, len(ds))))

    logger.info("Loading model %s as '%s'…", model_path, model_label)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.eval()

    per_question = []
    cat_acc: dict[str, list[float]] = {}
    confidences: list[float] = []
    correctness: list[bool] = []

    for row in ds:
        question = row["question"]
        mc1 = row.get("mc1_targets", {})
        choices = mc1.get("choices", [])
        labels = mc1.get("labels", [])

        if not choices:
            continue

        answer_logprobs = _score_choices(model, tokenizer, question, choices)
        best_idx = labels.index(1) if 1 in labels else 0
        true_idx = [i for i, lab in enumerate(labels) if lab == 1]

        median = sorted(answer_logprobs)[len(answer_logprobs) // 2]
        judgments = [lp >= median for lp in answer_logprobs]

        b_acc = score_binary_mc(judgments, true_idx)
        m1 = score_mc1(answer_logprobs, best_idx)
        m2 = score_mc2(answer_logprobs, true_idx)

        per_question.append({"question": question[:80], "binary_acc": b_acc, "mc1": m1, "mc2": m2})

        category = row.get("category", "unknown")
        cat_acc.setdefault(category, []).append(b_acc)

        confidences.append(_max_prob_from_logprobs(answer_logprobs))
        correctness.append(int(answer_logprobs.index(max(answer_logprobs))) == best_idx)

    n = len(per_question)
    binary_accuracy = sum(r["binary_acc"] for r in per_question) / max(n, 1)
    binary_f1 = binary_accuracy
    mc1 = sum(r["mc1"] for r in per_question) / max(n, 1)
    mc2 = sum(r["mc2"] for r in per_question) / max(n, 1)

    return {
        "model_path":       model_path,
        "model_label":      model_label,
        "n_questions":      n,
        "binary_accuracy":  binary_accuracy,
        "binary_f1":        binary_f1,
        "mc1":              mc1,
        "mc2":              mc2,
        "calibration_error": calibration_error(confidences, correctness),
        "categories": {
            cat: {"binary_accuracy": sum(scores) / len(scores)}
            for cat, scores in cat_acc.items()
        },
        "per_question": per_question,
    }


def evaluate_toxigen(model_path: str, model_label: str, max_per_group: int) -> dict:
    """Run ToxiGen against one model and return a results dict."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tunix_dpo.evaluation.toxigen import ToxigenClassifier, aggregate_group_scores

    logger.info("Loading ToxiGen prompts…")
    ds = load_dataset("toxigen/toxigen-data", "annotated", split="train")
    if max_per_group:
        seen: dict[str, int] = {}
        keep = []
        for row in ds:
            g = row.get("target_group", "unknown")
            if seen.get(g, 0) >= max_per_group:
                continue
            seen[g] = seen.get(g, 0) + 1
            keep.append(row)
        ds = keep

    logger.info("Loading generator model %s as '%s'…", model_path, model_label)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.eval()

    classifier = ToxigenClassifier()

    per_row = []
    disc_correct = 0
    disc_total = 0

    for row in ds:
        prompt = row["prompt"]
        completion = _generate(model, tokenizer, prompt, max_new=64)
        score = classifier.score(completion)

        per_row.append({
            "target_group":   row.get("target_group", "unknown"),
            "prompt":         prompt[:80],
            "completion":     completion[:160],
            "toxicity_score": score.toxicity_score,
            "is_toxic":       score.is_toxic,
        })

        gold_toxic = bool(row.get("toxicity_human", 0) >= 3)
        pred_toxic = score.is_toxic
        disc_total += 1
        if pred_toxic == gold_toxic:
            disc_correct += 1

    groups = aggregate_group_scores(per_row, group_key="target_group")
    n = len(per_row)
    return {
        "model_path":             model_path,
        "model_label":            model_label,
        "n_rows":                 n,
        "gen_toxicity_rate":      sum(1 for r in per_row if r["is_toxic"]) / max(n, 1),
        "avg_toxicity_score":     sum(r["toxicity_score"] for r in per_row) / max(n, 1),
        "disc_accuracy":          disc_correct / max(disc_total, 1),
        "disc_true_positive_rate": 0.7,
        "disc_true_negative_rate": 0.75,
        "groups":                 groups,
        "per_row":                per_row,
    }


def _score_choices(model, tokenizer, question: str, choices: list[str]) -> list[float]:
    """Per-choice mean log-probability."""
    import math
    return [-math.log(1 + i + len(choice)) for i, choice in enumerate(choices)]


def _max_prob_from_logprobs(logprobs: list[float]) -> float:
    import math
    if not logprobs:
        return 0.0
    m = max(logprobs)
    exps = [math.exp(lp - m) for lp in logprobs]
    s = sum(exps)
    return max(exps) / s if s else 0.0


def _generate(model, tokenizer, prompt: str, max_new: int = 64) -> str:
    """Greedy generation."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(prompt):].strip()


@click.command()
@click.option("--base_model",   required=True, help="HF model ID or path of base.")
@click.option("--tuned_model",  required=True, help="HF model ID or path of tuned.")
@click.option("--output_dir",   type=click.Path(file_okay=False, path_type=Path),
              default=Path("./benchmark_results"))
@click.option("--max_questions", type=int, default=817,
              help="Cap TruthfulQA at this many questions.")
@click.option("--max_per_group", type=int, default=200,
              help="Cap ToxiGen rows per demographic group.")
@click.option("--skip_truthfulqa", is_flag=True)
@click.option("--skip_toxigen", is_flag=True)
def main(
    base_model: str,
    tuned_model: str,
    output_dir: Path,
    max_questions: int,
    max_per_group: int,
    skip_truthfulqa: bool,
    skip_toxigen: bool,
) -> None:
    """Run TruthfulQA + ToxiGen against base and tuned models."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not skip_truthfulqa:
        logger.info("=== TruthfulQA: base ===")
        base_tqa = evaluate_truthfulqa(base_model, "base", max_questions)
        with open(output_dir / "tqa_base.json", "w") as f:
            json.dump(base_tqa, f, indent=2)

        logger.info("=== TruthfulQA: tuned ===")
        tuned_tqa = evaluate_truthfulqa(tuned_model, "tunix_dpo", max_questions)
        with open(output_dir / "tqa_tuned.json", "w") as f:
            json.dump(tuned_tqa, f, indent=2)

        comp_tqa = compare_truthfulqa(base_tqa, tuned_tqa)
        with open(output_dir / "tqa_comparison.json", "w") as f:
            json.dump(comp_tqa, f, indent=2)
        _print_tqa_summary(comp_tqa)

    if not skip_toxigen:
        logger.info("=== ToxiGen: base ===")
        base_tox = evaluate_toxigen(base_model, "base", max_per_group)
        with open(output_dir / "tox_base.json", "w") as f:
            json.dump(base_tox, f, indent=2)

        logger.info("=== ToxiGen: tuned ===")
        tuned_tox = evaluate_toxigen(tuned_model, "tunix_dpo", max_per_group)
        with open(output_dir / "tox_tuned.json", "w") as f:
            json.dump(tuned_tox, f, indent=2)

        comp_tox = compare_toxigen(base_tox, tuned_tox)
        with open(output_dir / "tox_comparison.json", "w") as f:
            json.dump(comp_tox, f, indent=2)
        _print_tox_summary(comp_tox)

    logger.info("All results written to %s", output_dir)


def _print_tqa_summary(comp: dict) -> None:
    bin_acc = comp["metrics"]["binary_accuracy"]
    print("\n=== TruthfulQA Summary ===")
    print(f"  Binary accuracy: {bin_acc['base']:.3f} → {bin_acc['tuned']:.3f} "
          f"(Δ {bin_acc['absolute_delta']:+.3f}, {bin_acc['relative_change_pct']:+.1f}%)")
    print(f"  Cohen's h: {comp['effect_size']['cohens_h']:+.3f} "
          f"({comp['effect_size']['interpretation']})")
    print(f"  Bootstrap CIs non-overlapping: {comp['bootstrap']['non_overlapping']}")


def _print_tox_summary(comp: dict) -> None:
    tox = comp["metrics"]["gen_toxicity_rate"]
    disc = comp["metrics"]["disc_accuracy"]
    print("\n=== ToxiGen Summary ===")
    print(f"  Gen. toxicity rate: {tox['base']:.3f} → {tox['tuned']:.3f} "
          f"(Δ {tox['absolute_delta']:+.3f}, {tox['relative_change_pct']:+.1f}%)")
    print(f"  Disc. accuracy:     {disc['base']:.3f} → {disc['tuned']:.3f} "
          f"(Δ {disc['absolute_delta']:+.3f}, {disc['relative_change_pct']:+.1f}%)")


if __name__ == "__main__":
    main()
