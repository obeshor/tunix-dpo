"""
Pure record constructors for the two output formats Tunix DPO produces:

- DPO records:  {prompt, chosen, rejected, metadata}
- RM records:   {prompt, response, label, metadata}
"""

from __future__ import annotations

from typing import Any


def format_dpo(
    prompt: str,
    chosen: str,
    rejected: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a single DPO training record."""
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": dict(metadata or {}),
    }


def format_rm(
    prompt: str,
    response: str,
    label: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a single Reward Model record."""
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label!r}")
    return {
        "prompt": prompt,
        "response": response,
        "label": int(label),
        "metadata": dict(metadata or {}),
    }
