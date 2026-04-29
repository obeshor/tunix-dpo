"""
Dataset and DataLoader for DPO training.

Tokenises the curated JSONL files, applies prompt-masking (labels=-100
over prompt tokens so the DPO loss is on the response only), and
reshapes batches to [n_devices, per_device_batch, seq_len] for pmap.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm

log = logging.getLogger(__name__)


class HHRLHFDataset(Dataset):
    """Tokenises a DPO JSONL file into padded numpy arrays.

    Each item contains:
        chosen_input_ids   : [max_len]
        chosen_labels      : [max_len]  — prompt tokens masked to -100
        rejected_input_ids : [max_len]
        rejected_labels    : [max_len]
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len   = max_len
        raw = load_dataset("json", data_files=str(path), split="train")
        log.info("Tokenising %s (%d rows)…", Path(path).name, len(raw))
        self._data = [self._encode(row) for row in tqdm(raw, leave=False)]

    def _encode(self, row: dict[str, Any]) -> dict[str, np.ndarray]:
        def tok(text: str) -> dict[str, np.ndarray]:
            return self.tokenizer(
                text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )

        chosen_enc   = tok(row["prompt"] + " " + row["chosen"])
        rejected_enc = tok(row["prompt"] + " " + row["rejected"])
        prompt_len   = len(self.tokenizer(row["prompt"])["input_ids"])

        def make_labels(ids: np.ndarray) -> np.ndarray:
            labels = ids.copy()
            labels[:, :prompt_len]                           = -100
            labels[ids == self.tokenizer.pad_token_id]       = -100
            return labels

        return {
            "chosen_input_ids":   chosen_enc["input_ids"][0],
            "chosen_labels":      make_labels(chosen_enc["input_ids"])[0],
            "rejected_input_ids": rejected_enc["input_ids"][0],
            "rejected_labels":    make_labels(rejected_enc["input_ids"])[0],
        }

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self._data[idx]


def numpy_collate(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack a list of dicts into a single dict of arrays."""
    return {k: np.stack([b[k] for b in batch]) for k in batch[0]}


def make_dataloader(
    dataset: HHRLHFDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Return a DataLoader that drops the last incomplete batch (required for pmap)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=numpy_collate,
        num_workers=num_workers,
        drop_last=True,
    )
