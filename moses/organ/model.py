from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from typing_extensions import override

from moses.organ.metrics_reward import MetricsReward

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rdkit import Chem

    from moses.utils import CharVocab, LstmOutT

    from .config import ORGANConfig


class Generator(nn.Module):
    def __init__(
        self,
        embedding_layer: nn.Embedding,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.embedding_layer = embedding_layer
        self.lstm_layer = nn.LSTM(  # type: ignore[no-untyped-call]
            embedding_layer.embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear_layer = nn.Linear(hidden_size, embedding_layer.num_embeddings)

    @override
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding_layer(x)
        packed_x = pack_padded_sequence(x, lengths, batch_first=True)
        lstm_out: LstmOutT = self.lstm_layer(packed_x, states)
        packed_x, states = lstm_out
        x, _ = pad_packed_sequence(packed_x, batch_first=True)
        x = self.linear_layer(x)

        return x, lengths, states


class Discriminator(nn.Module):
    def __init__(
        self,
        embedding_layer: nn.Embedding,
        convs: Sequence[tuple[int, int]],
        dropout: float = 0,
    ) -> None:
        super().__init__()

        self.embedding_layer = embedding_layer
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(1, f, kernel_size=(n, self.embedding_layer.embedding_dim)) for f, n in convs]
        )
        sum_filters = sum([f for f, _ in convs])
        self.highway_layer = nn.Linear(sum_filters, sum_filters)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(sum_filters, 1)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(x)
        x = x.unsqueeze(1)

        convs = [F.elu(conv_layer(x)).squeeze(3) for conv_layer in self.conv_layers]
        x_ls = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in convs]
        x = torch.cat(x_ls, dim=1)

        h: torch.Tensor = self.highway_layer(x)
        t = torch.sigmoid(h)
        x = t * F.elu(h) + (1 - t) * x
        x = self.dropout_layer(x)
        out: torch.Tensor = self.output_layer(x)

        return out


class ORGAN(nn.Module):
    def __init__(self, vocabulary: CharVocab, config: ORGANConfig) -> None:
        super().__init__()

        self.metrics_reward = MetricsReward(
            config.n_ref_subsample,
            config.rollouts,
            config.n_jobs,
            config.additional_rewards,
        )
        self.reward_weight = config.reward_weight

        self.vocabulary = vocabulary

        self.generator_embeddings = nn.Embedding(
            len(vocabulary), config.embedding_size, padding_idx=vocabulary.pad
        )
        self.discriminator_embeddings = nn.Embedding(
            len(vocabulary), config.embedding_size, padding_idx=vocabulary.pad
        )
        self.generator = Generator(
            self.generator_embeddings,
            config.hidden_size,
            config.num_layers,
            config.dropout,
        )
        self.discriminator = Discriminator(
            self.discriminator_embeddings,
            config.discriminator_layers,
            config.discriminator_dropout,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device  # type: ignore[no-any-return]

    def generator_forward(self, prevs: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        return self.generator(prevs, lens)  # type: ignore[no-any-return]

    def discriminator_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)  # type: ignore[no-any-return]

    # @override
    # def forward(self, *args, **kwargs):
    #     return self.sample(*args, **kwargs) # noqa: ERA001

    def string2tensor(self, string: str, device: str | torch.device = "model") -> torch.Tensor:
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        return torch.tensor(
            ids, dtype=torch.long, device=self.device if device == "model" else device
        )

    def tensor2string(self, tensor: torch.Tensor) -> str:
        ids = tensor.tolist()
        return self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

    def _proceed_sequences(
        self,
        prevs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
        max_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            n_sequences = prevs.shape[0]

            sequences: list[torch.Tensor] = []
            lengths = torch.zeros(n_sequences, dtype=torch.long, device=prevs.device)

            one_lens = torch.ones(n_sequences, dtype=torch.long, device="cpu")
            is_end = prevs.eq(self.vocabulary.eos).view(-1)

            for _ in range(max_len):
                outputs, _, states = self.generator(prevs, one_lens, states)
                probs = F.softmax(outputs, dim=-1).view(n_sequences, -1)
                currents = torch.multinomial(probs, 1)

                currents[is_end, :] = self.vocabulary.pad
                sequences.append(currents)
                lengths[~is_end] += 1

                is_end[currents.view(-1) == self.vocabulary.eos] = 1
                if is_end.sum() == n_sequences:
                    break

                prevs = currents

            cat_sequences = torch.cat(sequences, dim=-1)

        return cat_sequences, lengths

    def rollout(
        self,
        n_samples: int,
        n_rollouts: int,
        ref_smiles: Sequence[str],
        ref_mols: Sequence[Chem.Mol],
        max_len: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            sequences: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            lengths = torch.ones(n_samples, dtype=torch.long, device="cpu")

            one_lens = torch.ones(n_samples, dtype=torch.long, device="cpu")
            prevs = torch.empty(n_samples, 1, dtype=torch.long, device=self.device).fill_(
                self.vocabulary.bos
            )
            is_end = torch.zeros(n_samples, dtype=torch.bool, device="cpu")
            states: torch.Tensor | None = None

            sequences.append(prevs)

            for current_len in tqdm(range(max_len), desc="Rollout", leave=False):
                gen_output: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = self.generator(
                    prevs, one_lens, states
                )
                outputs, _, states = gen_output
                probs = F.softmax(outputs, dim=-1).view(n_samples, -1)
                currents = torch.multinomial(probs, 1)

                currents[is_end, :] = self.vocabulary.pad
                sequences.append(currents)
                lengths[~is_end] += 1

                rollout_prevs = currents[~is_end, :].repeat(n_rollouts, 1)
                rollout_states = (
                    states[0][:, ~is_end, :].repeat(1, n_rollouts, 1),
                    states[1][:, ~is_end, :].repeat(1, n_rollouts, 1),
                )
                rollout_sequences, rollout_lengths = self._proceed_sequences(
                    rollout_prevs, rollout_states, max_len - current_len
                )
                rollout_lengths = rollout_lengths.to("cpu")

                rollout_sequences = torch.cat(
                    [s[~is_end, :].repeat(n_rollouts, 1) for s in sequences] + [rollout_sequences],
                    dim=-1,
                )
                rollout_lengths += lengths[~is_end].repeat(n_rollouts)

                rollout_rewards = torch.sigmoid(self.discriminator(rollout_sequences).detach())

                if self.metrics_reward is not None and self.reward_weight > 0:  # type:ignore[redundant-expr]
                    strings = [
                        self.tensor2string(t[:l])
                        for t, l in zip(rollout_sequences, rollout_lengths)
                    ]
                    obj_rewards = torch.tensor(
                        self.metrics_reward(strings, ref_smiles, ref_mols),
                        device=rollout_rewards.device,
                    ).view(-1, 1)
                    rollout_rewards = (
                        rollout_rewards * (1 - self.reward_weight)
                        + obj_rewards * self.reward_weight
                    )

                current_rewards = torch.zeros(n_samples, device=self.device)

                current_rewards[~is_end] = rollout_rewards.view(n_rollouts, -1).mean(dim=0)
                rewards.append(current_rewards.view(-1, 1))

                is_end[currents.view(-1) == self.vocabulary.eos] = 1
                if is_end.sum() == n_samples:
                    break

                prevs = currents

            cat_sequences = torch.cat(sequences, dim=1)
            cat_rewards = torch.cat(rewards, dim=1)

        return cat_sequences, cat_rewards, lengths

    def sample_tensor(self, n: int, max_len: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
        prevs = torch.empty(n, 1, dtype=torch.long, device=self.device).fill_(self.vocabulary.bos)
        samples, lengths = self._proceed_sequences(prevs, None, max_len)

        samples = torch.cat([prevs, samples], dim=-1)
        lengths += 1  # type: ignore[assignment]

        return samples, lengths

    def sample(self, batch_n: int, max_len: int = 100) -> list[str]:
        samples, lengths = self.sample_tensor(batch_n, max_len)
        samples_ls = [t[:l] for t, l in zip(samples, lengths)]

        return [self.tensor2string(t) for t in samples_ls]
