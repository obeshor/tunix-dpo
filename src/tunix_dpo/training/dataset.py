"""
Dataset loading and collation for DPO training.

Reads a curated JSONL file produced by ``tunix-curate``, tokenises each pair
with the Gemma 3 1B IT tokenizer, and produces fixed-length batches with the
correct response-only masks for the DPO loss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class HHRLHFDataset:
    """Lazily loads JSONL records into memory and tokenises on demand."""

    def __init__(self, jsonl_path: str | Path, tokenizer, max_seq_len: int = 1024):
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(self.path) as f:
            self.records: list[dict[str, Any]] = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        rec = self.records[idx]
        return self._encode(rec["prompt"], rec["chosen"], rec["rejected"])

    def _encode(self, prompt: str, chosen: str, rejected: str) -> dict[str, np.ndarray]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        chosen_ids = self.tokenizer.encode(" " + chosen, add_special_tokens=False)
        rejected_ids = self.tokenizer.encode(" " + rejected, add_special_tokens=False)

        eos = self.tokenizer.eos_token_id
        if eos is not None:
            chosen_ids = chosen_ids + [eos]
            rejected_ids = rejected_ids + [eos]

        return {
            "chosen_ids":   self._pack(prompt_ids, chosen_ids),
            "chosen_mask":  self._mask(prompt_ids, chosen_ids),
            "rejected_ids": self._pack(prompt_ids, rejected_ids),
            "rejected_mask": self._mask(prompt_ids, rejected_ids),
        }

    def _pack(self, prompt_ids: list[int], response_ids: list[int]) -> np.ndarray:
        ids = (prompt_ids + response_ids)[: self.max_seq_len]
        pad_id = self.tokenizer.pad_token_id or 0
        ids = ids + [pad_id] * (self.max_seq_len - len(ids))
        return np.asarray(ids, dtype=np.int32)

    def _mask(self, prompt_ids: list[int], response_ids: list[int]) -> np.ndarray:
        """1 over response tokens, 0 over prompt and padding."""
        mask = np.zeros(self.max_seq_len, dtype=np.int32)
        start = min(len(prompt_ids), self.max_seq_len)
        end = min(start + len(response_ids), self.max_seq_len)
        mask[start:end] = 1
        return mask


def collate_batch(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack a list of encoded records into a single batch of arrays."""
    return {k: np.stack([r[k] for r in records], axis=0) for k in records[0]}


class DataLoader:
    """Minimal shuffled, batched iterator. No PyTorch dependency."""

    def __init__(
        self,
        dataset: HHRLHFDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.drop_last = drop_last

    def __iter__(self):
        n = len(self.dataset)
        idx = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(idx)
        for start in range(0, n, self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            yield collate_batch([self.dataset[int(i)] for i in batch_idx])

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
