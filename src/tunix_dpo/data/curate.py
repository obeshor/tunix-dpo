"""
CLI entry-point: curate HH-RLHF → DPO / RM JSONL files.

Usage
-----
    python -m tunix_dpo.data.curate --format dpo --output_dir ./data
    tunix-curate --format both --subsets helpful-base harmless-base
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from tunix_dpo.data.formatter import format_dpo, format_rm
from tunix_dpo.data.parser import is_valid_pair, parse_dialogue

log = logging.getLogger(__name__)

DATASET_ID      = "Anthropic/hh-rlhf"
DEFAULT_SUBSETS = ["helpful-base", "harmless-base"]
ALL_SUBSETS     = [
    "helpful-base",
    "helpful-online",
    "helpful-rejection-sampled",
    "harmless-base",
    "red-team-attempts",
]


# ── Core pipeline ─────────────────────────────────────────────────────────────

def process_subset(
    subset: str,
    split:  str,
    fmt:    str,   # "dpo" | "rm" | "both"
) -> dict[str, list[dict]]:
    """Load one subset/split, parse, filter, and return formatted records."""
    log.info("Loading  %s  subset=%r  split=%r", DATASET_ID, subset, split)
    ds = load_dataset(DATASET_ID, data_dir=subset, split=split)

    dpo_records: list[dict] = []
    rm_records:  list[dict] = []
    skipped = 0

    for row in tqdm(ds, desc=f"{subset}/{split}", leave=False):
        chosen_raw   = row.get("chosen",   "") or ""
        rejected_raw = row.get("rejected", "") or ""

        prompt, chosen_resp   = parse_dialogue(chosen_raw)
        _,      rejected_resp = parse_dialogue(rejected_raw)

        if not is_valid_pair(chosen_resp, rejected_resp):
            skipped += 1
            continue

        meta = {"subset": subset, "split": split}

        if fmt in ("dpo", "both"):
            dpo_records.append(format_dpo(prompt, chosen_resp, rejected_resp, meta))
        if fmt in ("rm", "both"):
            rm_records.append(format_rm(prompt, chosen_resp,   label=1, metadata=meta))
            rm_records.append(format_rm(prompt, rejected_resp, label=0, metadata=meta))

    n_valid = len(dpo_records) if fmt != "rm" else len(rm_records) // 2
    log.info("  Valid: %d  Skipped: %d", n_valid, skipped)

    result: dict[str, list[dict]] = {}
    if fmt in ("dpo", "both"):
        result["dpo"] = dpo_records
    if fmt in ("rm", "both"):
        result["rm"] = rm_records
    return result


# ── Writers ───────────────────────────────────────────────────────────────────

def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Wrote %d records → %s", len(records), path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Curate Anthropic HH-RLHF for DPO / RM.")
    parser.add_argument("--format",     choices=["dpo", "rm", "both"], default="dpo")
    parser.add_argument("--subsets",    nargs="+", default=None, metavar="SUBSET")
    parser.add_argument("--splits",     nargs="+", default=["train", "test"])
    parser.add_argument("--output_dir", default="./data")
    args = parser.parse_args(argv)

    subsets    = args.subsets or DEFAULT_SUBSETS
    output_dir = Path(args.output_dir)
    stats: dict = {}

    all_records: dict[str, dict[str, list]] = {"dpo": {}, "rm": {}}

    for subset in subsets:
        for split in args.splits:
            result = process_subset(subset, split, args.format)
            for fmt_key, records in result.items():
                all_records[fmt_key].setdefault(split, []).extend(records)
                stats.setdefault(subset, {}).setdefault(split, {})[fmt_key] = len(records)

    for fmt_key, split_map in all_records.items():
        for split, records in split_map.items():
            if records:
                write_jsonl(records, output_dir / fmt_key / f"{split}.jsonl")

    stats_path = output_dir / "curation_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info("Stats → %s", stats_path)


if __name__ == "__main__":
    main()
