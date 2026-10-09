"""Bounded, copy-safe cache for repeated prompt conditioning tensors."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable

import torch


class PromptEmbeddingCache:
    """Small LRU whose stored tensors cannot be changed by callers."""

    def __init__(self, max_entries: int = 4, max_bytes: int = 64 * 1024**2) -> None:
        if max_entries <= 0 or max_bytes <= 0:
            raise ValueError("prompt cache limits must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[
            Hashable, tuple[torch.Tensor, torch.Tensor, int]
        ] = OrderedDict()
        self.bytes = 0

    def get(self, key: Hashable) -> tuple[torch.Tensor, torch.Tensor] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        embeddings, mask, _ = entry
        return embeddings.clone(), mask.clone()

    def put(
        self, key: Hashable, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> None:
        cached_embeddings = embeddings.detach().clone()
        cached_mask = mask.detach().clone()
        size = (
            cached_embeddings.numel() * cached_embeddings.element_size()
            + cached_mask.numel() * cached_mask.element_size()
        )
        if size > self.max_bytes:
            return
        existing = self._entries.pop(key, None)
        if existing is not None:
            self.bytes -= existing[2]
        while self._entries and (
            len(self._entries) >= self.max_entries or self.bytes + size > self.max_bytes
        ):
            _, evicted = self._entries.popitem(last=False)
            self.bytes -= evicted[2]
        self._entries[key] = (cached_embeddings, cached_mask, size)
        self.bytes += size

    @property
    def entries(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self.bytes = 0
