"""
Output record constructors for DPO and Reward-Model formats.

Keeping formatters separate from I/O lets callers build records
without touching the file system — important for unit tests.
"""

from __future__ import annotations

from typing import Any


def format_dpo(
    prompt: str,
    chosen: str,
    rejected: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a TRL / Tunix DPO training record.

    Schema
    ------
    {
        "prompt":   str,   # conversation context ending with "Assistant:"
        "chosen":   str,   # preferred response
        "rejected": str,   # dis-preferred response
        "metadata": dict   # optional provenance
    }
    """
    record: dict[str, Any] = {
        "prompt":   prompt,
        "chosen":   chosen,
        "rejected": rejected,
    }
    if metadata:
        record["metadata"] = metadata
    return record


def format_rm(
    prompt: str,
    response: str,
    label: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a binary Reward-Model training record.

    Schema
    ------
    {
        "prompt":   str,
        "response": str,
        "label":    int,  # 1 = chosen / preferred, 0 = rejected
        "metadata": dict
    }
    """
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label!r}")
    record: dict[str, Any] = {
        "prompt":   prompt,
        "response": response,
        "label":    label,
    }
    if metadata:
        record["metadata"] = metadata
    return record
