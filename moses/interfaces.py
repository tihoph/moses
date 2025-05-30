from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from moses.utils import CharVocab, set_torch_seed_to_all_gens

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    from numpy.typing import NDArray

M = TypeVar("M", bound=nn.Module)


class MosesTrainer(ABC):
    @property
    def n_workers(self) -> int:
        n_workers = self.config.n_workers  # type: ignore[attr-defined]
        return n_workers if n_workers != 1 else 0

    def get_collate_device(self, model: M) -> str | torch.device:
        n_workers = self.n_workers
        return "cpu" if n_workers > 0 else model.device  # type: ignore[return-value]

    def get_dataloader(
        self,
        model: M,
        data: NDArray[np.str_] | Sequence[str] | Dataset[str],
        collate_fn: Callable[..., Any] | None = None,
        shuffle: bool = True,
    ) -> DataLoader[str]:
        if collate_fn is None:
            collate_fn = self.get_collate_fn(model)
        return DataLoader(
            data,  # type:ignore[arg-type]
            batch_size=self.config.n_batch,  # type: ignore[attr-defined]
            shuffle=shuffle,
            num_workers=self.n_workers,
            collate_fn=collate_fn,
            worker_init_fn=set_torch_seed_to_all_gens if self.n_workers > 0 else None,
        )

    def get_collate_fn(self, model: M) -> Callable[..., Any] | None:
        return None

    @abstractmethod
    def get_vocabulary(self, data: NDArray[np.str_] | Sequence[str]) -> CharVocab:
        pass

    @abstractmethod
    def fit(
        self,
        model: M,
        train_data: NDArray[np.str_] | Sequence[str],
        val_data: NDArray[np.str_] | Sequence[str] | None = None,
    ) -> M:
        pass
