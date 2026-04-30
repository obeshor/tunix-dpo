"""
Dialogue parsing utilities for the Anthropic HH-RLHF dataset.

All functions are pure (no I/O, no side effects).
"""

from __future__ import annotations

import re

# Captures separator between turns: the role word becomes a separate element
_TURN_RE = re.compile(r"\n\n(Human|Assistant):\s*", re.IGNORECASE)

MIN_RESPONSE_CHARS = 10


def parse_dialogue(raw_text: str) -> tuple[str, str]:
    """Split a raw HH-RLHF dialogue string into (prompt, final_response).

    The raw format::

        \\n\\nHuman: <turn>\\n\\nAssistant: <turn>...\\n\\nAssistant: <response>

    Returns
    -------
    prompt : str
        All turns up to (and including) the final ``\\n\\nAssistant:`` marker,
        so it is **identical** for both chosen and rejected completions —
        the key DPO invariant.
    response : str
        The final assistant turn only.
    """
    # split produces: ['', 'Human', 'content', 'Assistant', 'content', ...]
    parts = _TURN_RE.split(raw_text)

    # Drop leading empty string if present
    if parts and parts[0].strip() == "":
        parts = parts[1:]

    # Zip consecutive (role, content) pairs
    turns: list[dict[str, str]] = []
    i = 0
    while i + 1 < len(parts):
        role = parts[i].strip()
        content = parts[i + 1].strip()
        turns.append({"role": role, "content": content})
        i += 2

    if not turns:
        return raw_text.strip(), ""

    # The last turn must be the assistant's final response
    last = turns[-1]
    if last["role"].lower() != "assistant":
        return raw_text.strip(), ""

    response = last["content"]

    # Rebuild the prompt from all preceding turns
    prompt_parts: list[str] = []
    for turn in turns[:-1]:
        tag = "\n\nHuman: " if turn["role"].lower() == "human" else "\n\nAssistant: "
        prompt_parts.append(tag + turn["content"])
    prompt_parts.append("\n\nAssistant:")

    return "".join(prompt_parts).lstrip("\n"), response


def is_valid_pair(chosen: str, rejected: str) -> bool:
    """Return True when a chosen/rejected pair carries a usable preference signal."""
    if not chosen or not rejected:
        return False
    if chosen.strip() == rejected.strip():
        return False
    if len(chosen.strip()) < MIN_RESPONSE_CHARS:
        return False
    if len(rejected.strip()) < MIN_RESPONSE_CHARS:
        return False
    return True
