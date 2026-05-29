"""
HH-RLHF curation CLI — registered as ``tunix-curate``.

Reads the public Anthropic/hh-rlhf dataset from HuggingFace, applies the
parser and validation filters, and writes JSONL files in DPO and/or RM
format ready for downstream training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
from datasets import load_dataset

from tunix_dpo.data.formatter import format_dpo, format_rm
from tunix_dpo.data.parser import is_valid_pair, parse_dialogue

logger = logging.getLogger(__name__)

VALID_SUBSETS = ("helpful-base", "harmless-base", "helpful-online", "harmless-online")
VALID_SPLITS = ("train", "test")


def process_subset(
    subset: str,
    split: str,
    output_format: str,
) -> dict[str, list[dict[str, Any]]]:
    """Download (subset, split), apply parser/filter, return DPO/RM records."""
    if subset not in VALID_SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {VALID_SUBSETS}")
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {VALID_SPLITS}")
    if output_format not in {"dpo", "rm", "both"}:
        raise ValueError(f"Unknown format {output_format!r}; expected dpo|rm|both")

    rows = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split=split)

    dpo_records: list[dict[str, Any]] = []
    rm_records: list[dict[str, Any]] = []
    n_filtered_identical = 0
    n_filtered_short = 0

    for row in rows:
        chosen_raw = row.get("chosen", "")
        rejected_raw = row.get("rejected", "")
        if not chosen_raw or not rejected_raw:
            continue

        prompt_c, chosen = parse_dialogue(chosen_raw)
        prompt_r, rejected = parse_dialogue(rejected_raw)
        if prompt_c != prompt_r:
            continue

        if not is_valid_pair(chosen, rejected):
            if chosen.strip() == rejected.strip():
                n_filtered_identical += 1
            else:
                n_filtered_short += 1
            continue

        meta = {"subset": subset, "split": split}
        if output_format in {"dpo", "both"}:
            dpo_records.append(format_dpo(prompt_c, chosen, rejected, meta))
        if output_format in {"rm", "both"}:
            rm_records.append(format_rm(prompt_c, chosen, label=1, metadata=meta))
            rm_records.append(format_rm(prompt_r, rejected, label=0, metadata=meta))

    logger.info(
        "subset=%s split=%s kept=%d filtered_identical=%d filtered_short=%d",
        subset, split, len(dpo_records or rm_records),
        n_filtered_identical, n_filtered_short,
    )

    out: dict[str, list[dict[str, Any]]] = {}
    if output_format in {"dpo", "both"}:
        out["dpo"] = dpo_records
    if output_format in {"rm", "both"}:
        out["rm"] = rm_records
    return out


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write ``records`` as one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@click.command()
@click.option(
    "--format", "output_format",
    type=click.Choice(["dpo", "rm", "both"]),
    default="dpo",
    help="Output record format.",
)
@click.option(
    "--subsets",
    multiple=True,
    default=("helpful-base", "harmless-base"),
    help="HH-RLHF subsets to include (repeatable).",
)
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./data"),
    help="Where to write the JSONL files.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging.")
def main(output_format: str, subsets: tuple[str, ...], output_dir: Path, verbose: bool) -> None:
    """Curate the Anthropic HH-RLHF dataset into DPO / RM training files."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output_dir = Path(output_dir)
    all_dpo: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    all_rm: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}

    for subset in subsets:
        for split in VALID_SPLITS:
            logger.info("Processing subset=%s split=%s", subset, split)
            result = process_subset(subset, split, output_format)
            if "dpo" in result:
                all_dpo[split].extend(result["dpo"])
            if "rm" in result:
                all_rm[split].extend(result["rm"])

    if output_format in {"dpo", "both"}:
        for split in VALID_SPLITS:
            path = output_dir / "dpo" / f"{split}.jsonl"
            write_jsonl(all_dpo[split], path)
            logger.info("Wrote %d DPO records → %s", len(all_dpo[split]), path)

    if output_format in {"rm", "both"}:
        for split in VALID_SPLITS:
            path = output_dir / "rm" / f"{split}.jsonl"
            write_jsonl(all_rm[split], path)
            logger.info("Wrote %d RM records → %s", len(all_rm[split]), path)


if __name__ == "__main__":
    main()
