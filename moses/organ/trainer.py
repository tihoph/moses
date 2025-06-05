from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _Loss as Loss
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from typing_extensions import override

from moses.interfaces import MosesTrainer
from moses.utils import CharVocab, Logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from .config import ORGANConfig
    from .model import ORGAN


class PolicyGradientLoss(Loss):
    @override
    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        rewards: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(outputs, dim=2)
        items = torch.gather(log_probs, 2, targets.unsqueeze(2)) * rewards.unsqueeze(2)
        return (  # type: ignore[no-any-return]
            -sum([t[:l].sum() for t, l in zip(items, lengths)]) / lengths.sum().float()
        )


class ORGANTrainer(MosesTrainer):
    def __init__(self, config: ORGANConfig) -> None:
        self.config = config

    def generator_collate_fn(
        self, model: ORGAN
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
    def get_vocabulary(self, data: NDArray[np.str_] | Sequence[str]) -> CharVocab:
        return CharVocab.from_data(data)

    def _pretrain_generator_epoch(
        self,
        model: ORGAN,
        tqdm_data: tqdm[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        criterion: Loss,
        optimizer: Optimizer | None = None,
    ) -> dict[str, Any]:
        model.discriminator.eval()
        if optimizer is None:
            model.generator.eval()
        else:
            model.generator.train()

        postfix: dict[str, Any] = {"loss": 0, "running_loss": 0}

        for i, (prevs, nexts, lens) in enumerate(tqdm_data):
            prevs = prevs.to(model.device)
            nexts = nexts.to(model.device)
            lens = lens.to("cpu")

            outputs, _, _ = model.generator_forward(prevs, lens)

            loss: torch.Tensor = criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()  # type:ignore[no-untyped-call]
                optimizer.step()

            postfix["loss"] = loss.item()
            postfix["running_loss"] += (loss.item() - postfix["running_loss"]) / (i + 1)
            tqdm_data.set_postfix(postfix)


        postfix["mode"] = (
            "Pretrain: eval generator" if optimizer is None else "Pretrain: train generator"
        )
        return postfix

    def _pretrain_generator(
        self,
        model: ORGAN,
        train_loader: DataLoader[str],
        val_loader: DataLoader[str] | None = None,
        logger: Logger | None = None,
    ) -> None:
        device = model.device
        generator = model.generator
        criterion = nn.CrossEntropyLoss(ignore_index=model.vocabulary.pad)
        optimizer = torch.optim.Adam(generator.parameters(), lr=self.config.lr)

        generator.zero_grad()
        for epoch in range(self.config.generator_pretrain_epochs):
            tqdm_data = tqdm(train_loader, desc="Generator training (epoch #{})".format(epoch))
            postfix = self._pretrain_generator_epoch(model, tqdm_data, criterion, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc="Generator validation (epoch #{})".format(epoch))
                postfix = self._pretrain_generator_epoch(model, tqdm_data, criterion)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            if epoch % self.config.save_frequency == 0:
                generator = generator.to("cpu")
                torch.save(
                    generator.state_dict(),
                    self.config.model_save[:-3] + "_generator_{0:03d}.pt".format(epoch),
                )
                generator = generator.to(device)

    def _pretrain_discriminator_epoch(
        self,
        model: ORGAN,
        tqdm_data: tqdm[torch.Tensor],
        criterion: Loss,
        optimizer: Optimizer | None = None,
    ) -> dict[str, Any]:
        model.generator.eval()
        if optimizer is None:
            model.discriminator.eval()
        else:
            model.discriminator.train()

        postfix: dict[str, Any] = {"loss": 0, "running_loss": 0}

        for i, inputs_from_data in enumerate(tqdm_data):
            inputs_from_data = inputs_from_data.to(model.device)
            inputs_from_model, _ = model.sample_tensor(self.config.n_batch, self.config.max_length)

            targets = torch.zeros(self.config.n_batch, 1, device=model.device)
            outputs = model.discriminator_forward(inputs_from_model)
            loss: torch.Tensor = criterion(outputs, targets) / 2

            targets = torch.ones(inputs_from_data.shape[0], 1, device=model.device)
            outputs = model.discriminator_forward(inputs_from_data)
            loss += criterion(outputs, targets) / 2

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()  # type:ignore[no-untyped-call]
                optimizer.step()

            postfix["loss"] = loss.item()
            postfix["running_loss"] += (loss.item() - postfix["running_loss"]) / (i + 1)
            tqdm_data.set_postfix(postfix)


        postfix["mode"] = (
            "Pretrain: eval discriminator" if optimizer is None else "Pretrain: train discriminator"
        )
        return postfix

    def discriminator_collate_fn(self, model: ORGAN) -> Callable[[list[str]], torch.Tensor]:
        device = self.get_collate_device(model)

        def collate(data: list[str]) -> torch.Tensor:
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device) for string in data]
            return pad_sequence(tensors, batch_first=True, padding_value=model.vocabulary.pad)

        return collate

    def _pretrain_discriminator(
        self,
        model: ORGAN,
        train_loader: DataLoader[str],
        val_loader: DataLoader[str] | None = None,
        logger: Logger | None = None,
    ) -> None:
        device = model.device
        discriminator = model.discriminator
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.config.lr)

        discriminator.zero_grad()
        for epoch in range(self.config.discriminator_pretrain_epochs):
            tqdm_data = tqdm(train_loader, desc="Discriminator training (epoch #{})".format(epoch))
            postfix = self._pretrain_discriminator_epoch(model, tqdm_data, criterion, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(
                    val_loader,
                    desc="Discriminator validation (epoch #{})".format(epoch),
                )
                postfix = self._pretrain_discriminator_epoch(model, tqdm_data, criterion)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            if epoch % self.config.save_frequency == 0:
                discriminator = discriminator.to("cpu")
                torch.save(
                    discriminator.state_dict(),
                    self.config.model_save[:-3] + "_discriminator_{0:03d}.pt".format(epoch),
                )
                discriminator = discriminator.to(device)

    def _policy_gradient_iter(
        self,
        model: ORGAN,
        train_loader: DataLoader[str],
        criterion: Mapping[str, Loss],
        optimizer: Mapping[str, Optimizer],
        iter_: int,
    ) -> dict[str, Any]:
        if not self.ref_smiles or not self.ref_mols:
            raise RuntimeError("Reference smiles not set. Please set it before training.")

        smooth = self.config.pg_smooth_const if iter_ > 0 else 1

        # Generator
        gen_postfix: dict[str, Any] = {"generator_loss": 0, "smoothed_reward": 0}

        gen_tqdm = tqdm(
            range(self.config.generator_updates),
            desc="PG generator training (iter #{})".format(iter_),
        )
        for _ in gen_tqdm:
            model.eval()
            sequences, rewards, lengths = model.rollout(
                self.config.n_batch,
                self.config.rollouts,
                self.ref_smiles,
                self.ref_mols,
                self.config.max_length,
            )
            model.train()

            lengths, indices = torch.sort(lengths, descending=True)
            sequences = sequences[indices, ...]
            rewards = rewards[indices, ...]

            generator_outputs, lengths, _ = model.generator_forward(sequences[:, :-1], lengths - 1)
            generator_loss: torch.Tensor = criterion["generator"](
                generator_outputs, sequences[:, 1:], rewards, lengths
            )

            optimizer["generator"].zero_grad()
            generator_loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_value_(  # type: ignore[attr-defined]
                model.generator.parameters(), self.config.clip_grad
            )
            optimizer["generator"].step()

            gen_postfix["generator_loss"] += (
                generator_loss.item() - gen_postfix["generator_loss"]
            ) * smooth
            mean_episode_reward = torch.cat([t[:l] for t, l in zip(rewards, lengths)]).mean().item()
            gen_postfix["smoothed_reward"] += (
                mean_episode_reward - gen_postfix["smoothed_reward"]
            ) * smooth
            gen_tqdm.set_postfix(gen_postfix)

        # Discriminator
        discrim_postfix: dict[str, Any] = {"discrim-r_loss": 0}
        discrim_tqdm = tqdm(
            range(self.config.discriminator_updates),
            desc="PG discrim-r training (iter #{})".format(iter_),
        )
        for _ in discrim_tqdm:
            model.generator.eval()
            n_batches = (len(train_loader) + self.config.n_batch - 1) // self.config.n_batch
            sampled_batches = [
                model.sample_tensor(self.config.n_batch, self.config.max_length)[0]
                for _ in tqdm(range(n_batches), desc="Sampling batches", leave=False)
            ]

            for _ in tqdm(
                range(self.config.discriminator_epochs), desc="Discrim training", leave=False
            ):
                random.shuffle(sampled_batches)

                for inputs_from_model, inputs_from_data in zip(sampled_batches, train_loader):
                    inputs_from_data = inputs_from_data.to(model.device)

                    discrim_outputs = model.discriminator_forward(inputs_from_model)
                    discrim_targets = torch.zeros(len(discrim_outputs), 1, device=model.device)
                    discrim_fake_loss: torch.Tensor = (
                        criterion["discriminator"](discrim_outputs, discrim_targets) / 2
                    )

                    discrim_outputs = model.discriminator_forward(inputs_from_data)
                    discrim_targets = torch.ones(len(discrim_outputs), 1, device=model.device)
                    discrim_real_loss: torch.Tensor = (
                        criterion["discriminator"](discrim_outputs, discrim_targets) / 2
                    )
                    discrim_loss = discrim_fake_loss + discrim_real_loss

                    optimizer["discriminator"].zero_grad()
                    discrim_loss.backward()
                    optimizer["discriminator"].step()

                    discrim_postfix["discrim-r_loss"] += (
                        discrim_loss.item() - discrim_postfix["discrim-r_loss"]
                    ) * smooth

            discrim_tqdm.set_postfix(discrim_postfix)

        postfix = {**gen_postfix, **discrim_postfix}
        postfix["mode"] = "Policy Gradient (iter #{})".format(iter_)
        return postfix

    def _train_policy_gradient(
        self, model: ORGAN, train_loader: DataLoader[str], logger: Logger | None = None
    ) -> None:
        device = model.device

        criterion: dict[str, Loss] = {
            "generator": PolicyGradientLoss(),
            "discriminator": nn.BCEWithLogitsLoss(),
        }

        optimizer: dict[str, Optimizer] = {
            "generator": torch.optim.Adam(model.generator.parameters(), lr=self.config.lr),
            "discriminator": torch.optim.Adam(model.discriminator.parameters(), lr=self.config.lr),
        }

        model.zero_grad()
        for iter_ in range(self.config.pg_iters):
            postfix = self._policy_gradient_iter(model, train_loader, criterion, optimizer, iter_)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if iter_ % self.config.save_frequency == 0:
                model = model.to("cpu")
                torch.save(
                    model.state_dict(),
                    self.config.model_save[:-3] + "_{0:03d}.pt".format(iter_),
                )
                model = model.to(device)

    @override
    def fit(  # type: ignore[override]
        self,
        model: ORGAN,
        train_data: NDArray[np.str_] | Sequence[str],
        val_data: NDArray[np.str_] | Sequence[str] | None = None,
    ) -> ORGAN:
        logger = Logger() if self.config.log_file is not None else None

        # Generator
        gen_collate_fn = self.generator_collate_fn(model)
        gen_train_loader = self.get_dataloader(model, train_data, gen_collate_fn, shuffle=True)
        gen_val_loader = (
            None
            if val_data is None
            else self.get_dataloader(model, val_data, gen_collate_fn, shuffle=False)
        )
        self._pretrain_generator(model, gen_train_loader, gen_val_loader, logger)

        # Discriminator
        dsc_collate_fn = self.discriminator_collate_fn(model)
        dsc_train_loader = self.get_dataloader(model, train_data, dsc_collate_fn, shuffle=True)
        dsc_val_loader = (
            None
            if val_data is None
            else self.get_dataloader(model, val_data, dsc_collate_fn, shuffle=False)
        )
        self._pretrain_discriminator(model, dsc_train_loader, dsc_val_loader, logger)

        # Policy gradient
        self.ref_smiles, self.ref_mols = None, None
        if model.metrics_reward is not None:
            (self.ref_smiles, self.ref_mols) = model.metrics_reward.get_reference_data(train_data)

        pg_train_loader = dsc_train_loader
        self._train_policy_gradient(model, pg_train_loader, logger)

        del self.ref_smiles
        del self.ref_mols

        return model
