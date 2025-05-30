from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from typing_extensions import override

from moses.interfaces import MosesTrainer
from moses.utils import CharVocab, Logger

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    import numpy as np
    from numpy.typing import NDArray
    from torch.nn.modules.loss import _Loss as Loss
    from torch.nn.parameter import Parameter
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from .config import CharRNNConfig
    from .model import CharRNN


class CharRNNTrainer(MosesTrainer):
    def __init__(self, config: CharRNNConfig) -> None:
        self.config = config

    def _train_epoch(
        self,
        model: CharRNN,
        tqdm_data: tqdm[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        criterion: Loss,
        optimizer: Optimizer | None = None,
    ) -> dict[str, Any]:
        if optimizer is None:
            model.eval()
        else:
            model.train()

        postfix: dict[str, Any] = {"loss": 0, "running_loss": 0}

        for i, (prevs, nexts, lens) in enumerate(tqdm_data):
            prevs = prevs.to(model.device)
            nexts = nexts.to(model.device)
            lens = lens.to("cpu")

            outputs, _, _ = model(prevs, lens)

            loss: torch.Tensor = criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()

            postfix["loss"] = loss.item()
            postfix["running_loss"] += (loss.item() - postfix["running_loss"]) / (i + 1)
            tqdm_data.set_postfix(postfix)

        postfix["mode"] = "Eval" if optimizer is None else "Train"
        return postfix

    def _train(
        self,
        model: CharRNN,
        train_loader: DataLoader[str],
        val_loader: DataLoader[str] | None = None,
        logger: Logger | None = None,
    ) -> None:
        def get_params() -> Generator[Parameter]:
            return (p for p in model.parameters() if p.requires_grad)

        device = model.device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(get_params(), lr=self.config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.config.step_size, self.config.gamma)

        model.zero_grad()
        for epoch in range(self.config.train_epochs):
            tqdm_data = tqdm(train_loader, desc="Training (epoch #{})".format(epoch))
            postfix = self._train_epoch(model, tqdm_data, criterion, optimizer)

            scheduler.step()

            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc="Validation (epoch #{})".format(epoch))
                postfix = self._train_epoch(model, tqdm_data, criterion)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            if (self.config.model_save is not None) and (epoch % self.config.save_frequency == 0):
                model = model.to("cpu")
                torch.save(
                    model.state_dict(),
                    self.config.model_save[:-3] + "_{0:03d}.pt".format(epoch),
                )
                model = model.to(device)

    @override
    def get_vocabulary(self, data: NDArray[np.str_] | Sequence[str]) -> CharVocab:
        return CharVocab.from_data(data)

    @override
    def get_collate_fn(
        self,
        model: CharRNN,  # type: ignore[override]
    ) -> Callable[[list[str]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        device = self.get_collate_device(model)

        def collate(data: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device) for string in data]

            pad = model.vocabulary.pad
            prevs = pad_sequence([t[:-1] for t in tensors], batch_first=True, padding_value=pad)
            nexts = pad_sequence([t[1:] for t in tensors], batch_first=True, padding_value=pad)
            lens = torch.tensor([len(t) - 1 for t in tensors], dtype=torch.long, device=device)
            return prevs, nexts, lens

        return collate

    @override
    def fit(  # type: ignore[override]
        self,
        model: CharRNN,
        train_data: NDArray[np.str_] | Sequence[str],
        val_data: NDArray[np.str_] | Sequence[str] | None = None,
    ) -> CharRNN:
        logger = Logger() if self.config.log_file is not None else None

        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = (
            None if val_data is None else self.get_dataloader(model, val_data, shuffle=False)
        )

        self._train(model, train_loader, val_loader, logger)
        return model
