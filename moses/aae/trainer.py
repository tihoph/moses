from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from typing_extensions import override

from moses.interfaces import MosesTrainer
from moses.utils import CharVocab, Logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray
    from torch.nn.modules.loss import _Loss as Loss
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from .config import AAEConfig
    from .model import AAE

__all__ = ["AAETrainer"]


class AAETrainer(MosesTrainer):
    def __init__(self, config: AAEConfig) -> None:
        self.config = config

    def _pretrain_epoch(
        self,
        model: AAE,
        tqdm_data: tqdm[
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ]
        ],
        criterion: Loss,
        optimizer: Optimizer | None = None,
    ) -> dict[str, Any]:
        if optimizer is None:
            model.eval()
        else:
            model.train()

        postfix: dict[str, Any] = {
            "pretraining_loss": 0,
        }

        for i, (enc_inputs, dec_inputs, dec_targets) in enumerate(tqdm_data):
            encoder_inputs = (enc_inputs[0].to(model.device), enc_inputs[1].to("cpu"))
            decoder_inputs = (dec_inputs[0].to(model.device), dec_inputs[1].to("cpu"))
            decoder_targets = (
                dec_targets[0].to(model.device),
                dec_targets[1].to("cpu"),
            )

            latent_codes = model.encoder_forward(*encoder_inputs)
            decoder_outputs, decoder_output_lengths, _ = model.decoder_forward(
                *decoder_inputs, latent_codes, is_latent_states=True
            )

            cat_decoder_outputs = torch.cat(
                [t[:l] for t, l in zip(decoder_outputs, decoder_output_lengths)], dim=0
            )
            cat_decoder_targets = torch.cat([t[:l] for t, l in zip(*decoder_targets)], dim=0)

            loss: torch.Tensor = criterion(cat_decoder_outputs, cat_decoder_targets)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()  # type:ignore[no-untyped-call]
                optimizer.step()

            postfix["pretraining_loss"] += (loss.item() - postfix["pretraining_loss"]) / (i + 1)

            tqdm_data.set_postfix(postfix)

        postfix["mode"] = "Pretraining:Eval" if optimizer is None else "Pretraining:Train"

        return postfix

    def _pretrain(
        self,
        model: AAE,
        train_loader: DataLoader[str],
        val_loader: DataLoader[str] | None = None,
        logger: Logger | None = None,
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()),
            lr=self.config.lr,
        )
        device = model.device

        model.zero_grad()
        for epoch in range(self.config.pretrain_epochs):
            tqdm_data = tqdm(train_loader, desc="Pretraining (epoch #{})".format(epoch))

            postfix = self._pretrain_epoch(model, tqdm_data, criterion, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(
                    val_loader, desc="Pretraining validation (epoch #{})".format(epoch)
                )
                self._pretrain_epoch(model, tqdm_data, criterion)

            if epoch % self.config.save_frequency == 0:
                model = model.to("cpu")
                torch.save(
                    model.state_dict(),
                    self.config.model_save[:-3] + "_pretraining_{0:03d}.pt".format(epoch),
                )
                model = model.to(device)

    def _train_epoch(
        self,
        model: AAE,
        tqdm_data: tqdm[
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ]
        ],
        criterions: Mapping[str, Loss],
        optimizers: Mapping[str, Optimizer] | None = None,
    ) -> dict[str, Any]:
        if optimizers is None:
            model.eval()
        else:
            model.train()

        postfix: dict[str, Any] = {
            "autoencoder_loss": 0,
            "generator_loss": 0,
            "discriminator_loss": 0,
        }

        for i, (enc_inputs, dec_inputs, dec_targets) in enumerate(tqdm_data):
            encoder_inputs = (enc_inputs[0].to(model.device), enc_inputs[1].to("cpu"))
            decoder_inputs = (dec_inputs[0].to(model.device), dec_inputs[1].to("cpu"))
            decoder_targets = (
                dec_targets[0].to(model.device),
                dec_targets[1].to("cpu"),
            )

            latent_codes = model.encoder_forward(*encoder_inputs)
            decoder_outputs, decoder_output_lengths, _ = model.decoder_forward(
                *decoder_inputs, latent_codes, is_latent_states=True
            )
            discriminator_outputs = model.discriminator_forward(latent_codes)
            cat_decoder_outputs = torch.cat(
                [t[:l] for t, l in zip(decoder_outputs, decoder_output_lengths)], dim=0
            )
            cat_decoder_targets = torch.cat([t[:l] for t, l in zip(*decoder_targets)], dim=0)

            autoencoder_loss: torch.Tensor
            generator_loss: torch.Tensor
            discriminator_loss: torch.Tensor
            if i % (self.config.discriminator_steps + 1) == 0:
                autoencoder_loss = criterions["autoencoder"](
                    cat_decoder_outputs, cat_decoder_targets
                )
                discriminator_targets = torch.ones(latent_codes.shape[0], 1, device=model.device)
                generator_loss = criterions["discriminator"](
                    discriminator_outputs, discriminator_targets
                )
                total_loss = autoencoder_loss + generator_loss

                postfix["autoencoder_loss"] = (
                    postfix["autoencoder_loss"] * (i // 2) + autoencoder_loss.item()
                ) / (i // 2 + 1)
                postfix["generator_loss"] = (
                    postfix["generator_loss"] * (i // 2) + generator_loss.item()
                ) / (i // 2 + 1)
            else:
                discriminator_targets = torch.zeros(latent_codes.shape[0], 1, device=model.device)
                generator_loss = criterions["discriminator"](
                    discriminator_outputs, discriminator_targets
                )

                discriminator_inputs = model.sample_latent(latent_codes.shape[0])
                discriminator_outputs = model.discriminator(discriminator_inputs)
                discriminator_targets = torch.ones(latent_codes.shape[0], 1, device=model.device)
                discriminator_loss = criterions["discriminator"](
                    discriminator_outputs, discriminator_targets
                )
                total_loss = (generator_loss + discriminator_loss) / 2
                postfix["discriminator_loss"] = (
                    postfix["discriminator_loss"] * (i // 2) + total_loss.item()
                ) / (i // 2 + 1)

            if optimizers is not None:
                optimizers["autoencoder"].zero_grad()
                optimizers["discriminator"].zero_grad()
                total_loss.backward()
                for parameter in model.parameters():
                    parameter.grad.clamp_(-5, 5)
                if i % (self.config.discriminator_steps + 1) == 0:
                    optimizers["autoencoder"].step()
                else:
                    optimizers["discriminator"].step()

            tqdm_data.set_postfix(postfix)

        postfix["mode"] = "Eval" if optimizers is None else "Train"
        return postfix

    def _train(
        self,
        model: AAE,
        train_loader: DataLoader[str],
        val_loader: DataLoader[str] | None = None,
        logger: Logger | None = None,
    ) -> None:
        criterions: dict[str, Loss] = {
            "autoencoder": nn.CrossEntropyLoss(),
            "discriminator": nn.BCEWithLogitsLoss(),
        }

        optimizers: dict[str, Optimizer] = {
            "autoencoder": torch.optim.Adam(
                list(model.encoder.parameters()) + list(model.decoder.parameters()),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            ),
            "discriminator": torch.optim.Adam(
                model.discriminator.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            ),
        }
        schedulers = {
            k: torch.optim.lr_scheduler.StepLR(v, self.config.step_size, self.config.gamma)
            for k, v in optimizers.items()
        }
        device = model.device

        model.zero_grad()
        for epoch in range(self.config.train_epochs):
            tqdm_data = tqdm(train_loader, desc="Training (epoch #{})".format(epoch))

            postfix = self._train_epoch(model, tqdm_data, criterions, optimizers)

            for scheduler in schedulers.values():
                scheduler.step()

            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc="Validation (epoch #{})".format(epoch))
                postfix = self._train_epoch(model, tqdm_data, criterions)
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
    def get_collate_fn(  # type: ignore[override]
        self,
        model: AAE,
    ) -> Callable[
        [list[str]],
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
    ]:
        device = self.get_collate_device(model)

        def collate(
            data: list[str],
        ) -> tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ]:
            data.sort(key=lambda x: len(x), reverse=True)

            tensors = [model.string2tensor(string, device=device) for string in data]
            lengths = torch.tensor([len(t) for t in tensors], dtype=torch.long, device=device)

            encoder_inputs = pad_sequence(
                tensors, batch_first=True, padding_value=model.vocabulary.pad
            )
            encoder_input_lengths = lengths - 2

            decoder_inputs = pad_sequence(
                [t[:-1] for t in tensors],
                batch_first=True,
                padding_value=model.vocabulary.pad,
            )
            decoder_input_lengths = lengths - 1

            decoder_targets = pad_sequence(
                [t[1:] for t in tensors],
                batch_first=True,
                padding_value=model.vocabulary.pad,
            )
            decoder_target_lengths = lengths - 1

            return (  # type: ignore[return-value]
                (encoder_inputs, encoder_input_lengths),
                (decoder_inputs, decoder_input_lengths),
                (decoder_targets, decoder_target_lengths),
            )

        return collate

    @override
    def fit(  # type: ignore[override]
        self,
        model: AAE,
        train_data: NDArray[np.str_] | Sequence[str],
        val_data: NDArray[np.str_] | Sequence[str] | None = None,
    ) -> AAE:
        logger = Logger() if self.config.log_file is not None else None

        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = (
            None if val_data is None else self.get_dataloader(model, val_data, shuffle=False)
        )

        self._pretrain(model, train_loader, val_loader, logger)
        self._train(model, train_loader, val_loader, logger)
        return model
