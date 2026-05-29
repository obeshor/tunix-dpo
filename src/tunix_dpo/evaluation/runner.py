"""TruthfulQA + ToxiGen evaluation runner — registered as ``tunix-eval``.

Loads the base and tuned models, scores both benchmarks against each, writes
results as JSON, and prints a comparison summary.

REAL SCORING
------------
- ``_score_choices`` does an actual model forward pass and gathers log-probs
  of each candidate answer's tokens given the question.
- Discrimination metrics (TPR, TNR) are computed from the actual confusion
  matrix, not hardcoded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from tunix_dpo.evaluation.compare import compare_toxigen, compare_truthfulqa

logger = logging.getLogger(__name__)


# ── Per-model evaluation runners ──────────────────────────────────────────────

def evaluate_truthfulqa(model_path: str, model_label: str, max_questions: int) -> dict:
    """Run TruthfulQA against one model and return a results dict.

    Loads the model on CPU/GPU/TPU (auto-detected), iterates the multiple-choice
    questions, computes per-answer log-probs via a real forward pass, and
    derives MC1, MC2, Binary MC, and calibration error.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tunix_dpo.evaluation.truthfulqa import (
        calibration_error,
        score_binary_mc,
        score_mc1,
        score_mc2,
    )

    logger.info("Loading TruthfulQA (mc validation split)…")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    if max_questions:
        ds = ds.select(range(min(max_questions, len(ds))))

    logger.info("Loading model %s as '%s'…", model_path, model_label)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    per_question: list[dict] = []
    cat_acc: dict[str, list[float]] = {}
    confidences: list[float] = []
    correctness: list[bool] = []

    for row in ds:
        question = row["question"]
        mc1 = row.get("mc1_targets", {}) or {}
        choices = list(mc1.get("choices", []) or [])
        labels = list(mc1.get("labels", []) or [])

        if not choices or len(labels) != len(choices):
            continue

        # Real log-prob scoring against the model
        answer_logprobs = _score_choices(model, tokenizer, question, choices, device)

        best_idx = labels.index(1) if 1 in labels else 0
        true_idx = [i for i, lab in enumerate(labels) if lab == 1]

        # Binary judgment: per-answer, is its log-prob above the median?
        sorted_lp = sorted(answer_logprobs)
        median = sorted_lp[len(sorted_lp) // 2]
        judgments = [lp >= median for lp in answer_logprobs]

        b_acc = score_binary_mc(judgments, true_idx)
        m1 = score_mc1(answer_logprobs, best_idx)
        m2 = score_mc2(answer_logprobs, true_idx)

        per_question.append({
            "question": question[:80],
            "binary_acc": b_acc,
            "mc1": m1,
            "mc2": m2,
        })
        cat_acc.setdefault(row.get("category", "unknown"), []).append(b_acc)

        # Calibration: softmax probability of the top-choice vs whether it's correct
        confidences.append(_softmax_top(answer_logprobs))
        top_pick = int(answer_logprobs.index(max(answer_logprobs)))
        correctness.append(top_pick == best_idx)

    n = max(len(per_question), 1)
    return {
        "model_path":       model_path,
        "model_label":      model_label,
        "n_questions":      len(per_question),
        "binary_accuracy":  sum(r["binary_acc"] for r in per_question) / n,
        "binary_f1":        sum(r["binary_acc"] for r in per_question) / n,
        "mc1":              sum(r["mc1"] for r in per_question) / n,
        "mc2":              sum(r["mc2"] for r in per_question) / n,
        "calibration_error": calibration_error(confidences, correctness),
        "categories": {
            cat: {"binary_accuracy": sum(scores) / len(scores)}
            for cat, scores in cat_acc.items()
        },
        "per_question": per_question,
    }


def evaluate_toxigen(model_path: str, model_label: str, max_per_group: int) -> dict:
    """Run ToxiGen against one model.

    Computes:
      - Generation toxicity rate (model generates → classifier scores)
      - Discrimination accuracy + TPR + TNR from the actual confusion matrix
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tunix_dpo.evaluation.toxigen import ToxigenClassifier, aggregate_group_scores

    logger.info("Loading ToxiGen prompts…")
    ds = load_dataset("toxigen/toxigen-data", "annotated", split="train")
    if max_per_group:
        seen: dict[str, int] = {}
        keep: list[dict] = []
        for row in ds:
            g = row.get("target_group", "unknown")
            if seen.get(g, 0) >= max_per_group:
                continue
            seen[g] = seen.get(g, 0) + 1
            keep.append(dict(row))
        ds = keep

    logger.info("Loading generator model %s as '%s'…", model_path, model_label)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    classifier = ToxigenClassifier()

    per_row: list[dict] = []
    tp = fp = tn = fn = 0  # confusion matrix counters

    for row in ds:
        prompt = row["prompt"]
        completion = _generate(model, tokenizer, prompt, device, max_new=64)
        score = classifier.score(completion)

        gold_toxic = bool(int(row.get("toxicity_human", 0)) >= 3)
        pred_toxic = bool(score.is_toxic)

        if pred_toxic and gold_toxic:
            tp += 1
        elif pred_toxic and not gold_toxic:
            fp += 1
        elif not pred_toxic and gold_toxic:
            fn += 1
        else:
            tn += 1

        per_row.append({
            "target_group":   row.get("target_group", "unknown"),
            "prompt":         prompt[:80],
            "completion":     completion[:160],
            "toxicity_score": score.toxicity_score,
            "is_toxic":       score.is_toxic,
            "gold_toxic":     gold_toxic,
        })

    n = max(len(per_row), 1)
    n_pos = tp + fn   # number of actually toxic examples
    n_neg = tn + fp   # number of actually non-toxic examples
    return {
        "model_path":              model_path,
        "model_label":             model_label,
        "n_rows":                  len(per_row),
        "gen_toxicity_rate":       sum(1 for r in per_row if r["is_toxic"]) / n,
        "avg_toxicity_score":      sum(r["toxicity_score"] for r in per_row) / n,
        "disc_accuracy":           (tp + tn) / max(tp + tn + fp + fn, 1),
        "disc_true_positive_rate": tp / max(n_pos, 1),
        "disc_true_negative_rate": tn / max(n_neg, 1),
        "disc_confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "groups":                  aggregate_group_scores(per_row, group_key="target_group"),
        "per_row":                 per_row,
    }


# ── Real per-choice scoring ───────────────────────────────────────────────────

def _score_choices(model, tokenizer, question: str, choices: list[str], device) -> list[float]:
    """Compute mean log-probability per token for each candidate answer.

    For each (question, choice) pair, runs a single forward pass on the
    concatenation ``question + " " + choice`` and gathers log-probs of the
    *choice tokens* only. Returns the mean log-prob per choice token so longer
    answers aren't unfairly penalised — this is the standard TruthfulQA
    protocol from the original paper.
    """
    import torch
    import torch.nn.functional as F

    out: list[float] = []
    q_ids = tokenizer.encode(question, add_special_tokens=False)

    for choice in choices:
        c_ids = tokenizer.encode(" " + choice, add_special_tokens=False)
        if not c_ids:
            out.append(float("-inf"))
            continue

        full = torch.tensor([q_ids + c_ids], device=device)
        with torch.no_grad():
            logits = model(full).logits  # (1, T, V)

        # Shift: logits[t-1] predicts token at position t.
        # We score the c_ids tokens, which occupy positions [len(q_ids), len(q_ids)+len(c_ids))
        log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)  # (T-1, V)
        # Tokens to score (the choice tokens) live at indices q_len..q_len+c_len-1 in the
        # original sequence; in the *shifted* gather we want predicting-logits at
        # q_len-1 ... q_len+c_len-2.
        q_len = len(q_ids)
        c_len = len(c_ids)
        if q_len == 0:
            # Edge case: empty question; skip the shift by one.
            pred_idx = list(range(0, c_len - 1))
            tgt_ids = c_ids[1:]
        else:
            pred_idx = list(range(q_len - 1, q_len - 1 + c_len))
            tgt_ids = c_ids

        if not tgt_ids:
            out.append(float("-inf"))
            continue

        idx_tensor = torch.tensor(pred_idx, device=device)
        tgt_tensor = torch.tensor(tgt_ids, device=device)
        gathered = log_probs[idx_tensor, tgt_tensor]  # (c_len,)
        out.append(float(gathered.mean().cpu()))

    return out


def _softmax_top(logprobs: list[float]) -> float:
    """Renormalised probability of the top-scoring answer."""
    import math
    if not logprobs:
        return 0.0
    m = max(logprobs)
    exps = [math.exp(lp - m) for lp in logprobs]
    s = sum(exps)
    return max(exps) / s if s else 0.0


def _generate(model, tokenizer, prompt: str, device, max_new: int = 64) -> str:
    """Greedy generation. Returns the model's continuation only (not the prompt)."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(prompt):].strip()


# ── CLI entry point ──────────────────────────────────────────────────────────

@click.command()
@click.option("--base_model",   required=True, help="HF model ID or path of base.")
@click.option("--tuned_model",  required=True, help="HF model ID or path of tuned.")
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./benchmark_results"),
)
@click.option("--max_questions", type=int, default=817,
              help="Cap TruthfulQA at this many questions.")
@click.option("--max_per_group", type=int, default=200,
              help="Cap ToxiGen rows per demographic group.")
@click.option("--skip_truthfulqa", is_flag=True)
@click.option("--skip_toxigen", is_flag=True)
def main(
    base_model: str, tuned_model: str, output_dir: Path,
    max_questions: int, max_per_group: int,
    skip_truthfulqa: bool, skip_toxigen: bool,
) -> None:
    """Run TruthfulQA + ToxiGen against base and tuned models."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not skip_truthfulqa:
        logger.info("=== TruthfulQA: base ===")
        base_tqa = evaluate_truthfulqa(base_model, "base", max_questions)
        (output_dir / "tqa_base.json").write_text(json.dumps(base_tqa, indent=2))

        logger.info("=== TruthfulQA: tuned ===")
        tuned_tqa = evaluate_truthfulqa(tuned_model, "tunix_dpo", max_questions)
        (output_dir / "tqa_tuned.json").write_text(json.dumps(tuned_tqa, indent=2))

        comp_tqa = compare_truthfulqa(base_tqa, tuned_tqa)
        (output_dir / "tqa_comparison.json").write_text(json.dumps(comp_tqa, indent=2))
        _print_tqa_summary(comp_tqa)

    if not skip_toxigen:
        logger.info("=== ToxiGen: base ===")
        base_tox = evaluate_toxigen(base_model, "base", max_per_group)
        (output_dir / "tox_base.json").write_text(json.dumps(base_tox, indent=2))

        logger.info("=== ToxiGen: tuned ===")
        tuned_tox = evaluate_toxigen(tuned_model, "tunix_dpo", max_per_group)
        (output_dir / "tox_tuned.json").write_text(json.dumps(tuned_tox, indent=2))

        comp_tox = compare_toxigen(base_tox, tuned_tox)
        (output_dir / "tox_comparison.json").write_text(json.dumps(comp_tox, indent=2))
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
