from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_  # type: ignore[attr-defined]
from tqdm.auto import tqdm
from typing_extensions import override

from moses.interfaces import MosesTrainer
from moses.utils import CircularBuffer, Logger, OneHotVocab
from moses.vae.misc import CosineAnnealingLRWithRestart, KLAnnealer

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    import numpy as np
    from numpy.typing import NDArray
    from torch.nn.parameter import Parameter
    from torch.utils.data import DataLoader

    from .config import VAEConfig
    from .model import VAE


class VAETrainer(MosesTrainer):
    def __init__(self, config: VAEConfig) -> None:
        self.config = config

    @override
    def get_vocabulary(self, data: NDArray[np.str_] | Sequence[str]) -> OneHotVocab:
        return OneHotVocab.from_data(data)

    @override
    def get_collate_fn(self, model: VAE) -> Callable[[list[str]], list[torch.Tensor]]:  # type: ignore[override]
        device = self.get_collate_device(model)

        def collate(data: list[str]) -> list[torch.Tensor]:
            data.sort(key=len, reverse=True)
            return [model.string2tensor(string, device=device) for string in data]

        return collate

    def _train_epoch(
        self,
        model: VAE,
        epoch: int,
        tqdm_data: tqdm[Sequence[torch.Tensor]],
        kl_weight: float,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        if optimizer is None:
            model.eval()
        else:
            model.train()

        kl_loss_values = CircularBuffer(self.config.n_last)
        recon_loss_values = CircularBuffer(self.config.n_last)
        loss_values = CircularBuffer(self.config.n_last)

        result: dict[str, Any] | None = None
        for input_batch in tqdm_data:
            input_batch = tuple(data.to(model.device) for data in input_batch)

            # Forward
            var_output: tuple[torch.Tensor, torch.Tensor] = model(input_batch)
            kl_loss, recon_loss = var_output
            loss: torch.Tensor = kl_weight * kl_loss + recon_loss  # type: ignore[assignment]

            # Backward
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                clip_grad_norm_(self.get_optim_params(model), self.config.clip_grad)
                optimizer.step()

            # Log
            kl_loss_values.add(kl_loss.item())
            recon_loss_values.add(recon_loss.item())
            loss_values.add(loss.item())
            lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0

            # Update tqdm
            kl_loss_value = kl_loss_values.mean()
            recon_loss_value = recon_loss_values.mean()
            loss_value = loss_values.mean()
            postfix = [
                f"loss={loss_value:.5f}",
                f"(kl={kl_loss_value:.5f}",
                f"recon={recon_loss_value:.5f})",
                f"klw={kl_weight:.5f} lr={lr:.5f}",
            ]
            tqdm_data.set_postfix_str(" ".join(postfix))

            result = {
                "epoch": epoch,
                "kl_weight": kl_weight,
                "lr": lr,
                "kl_loss": kl_loss_value,
                "recon_loss": recon_loss_value,
                "loss": loss_value,
                "mode": "Eval" if optimizer is None else "Train",
            }
        if result is None:
            raise RuntimeError("No result was returned")
        return result

    def get_optim_params(self, model: VAE) -> Generator[Parameter]:
        return (p for p in model.vae.parameters() if p.requires_grad)

    def _train(
        self,
        model: VAE,
        train_loader: DataLoader[str],
        val_loader: DataLoader[str] | None = None,
        logger: Logger | None = None,
    ) -> None:
        device = model.device
        n_epoch = self._n_epoch()

        optimizer = optim.Adam(self.get_optim_params(model), lr=self.config.lr_start)
        kl_annealer = KLAnnealer(n_epoch, self.config)
        lr_annealer = CosineAnnealingLRWithRestart(optimizer, self.config)

        model.zero_grad()
        for epoch in range(n_epoch):
            # Epoch start
            kl_weight = kl_annealer(epoch)
            tqdm_data = tqdm(train_loader, desc="Training (epoch #{})".format(epoch))
            postfix = self._train_epoch(model, epoch, tqdm_data, kl_weight, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc="Validation (epoch #{})".format(epoch))
                postfix = self._train_epoch(model, epoch, tqdm_data, kl_weight)
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

            # Epoch end
            lr_annealer.step()

    @override
    def fit(  # type: ignore[override]
        self,
        model: VAE,
        train_data: NDArray[np.str_] | Sequence[str],
        val_data: NDArray[np.str_] | Sequence[str] | None = None,
    ) -> VAE:
        logger = Logger() if self.config.log_file is not None else None

        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = (
            None if val_data is None else self.get_dataloader(model, val_data, shuffle=False)
        )

        self._train(model, train_loader, val_loader, logger)
        return model

    def _n_epoch(self) -> int:
        return sum(
            self.config.lr_n_period * (self.config.lr_n_mult**i)
            for i in range(self.config.lr_n_restarts)
        )
