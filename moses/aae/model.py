from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Sequence

    from moses.utils import CharVocab, LstmOutT

    from .config import AAEConfig


class Encoder(nn.Module):
    def __init__(
        self,
        embedding_layer: nn.Embedding,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        latent_size: int,
    ) -> None:
        super().__init__()

        self.embedding_layer = embedding_layer
        self.lstm_layer = nn.LSTM(  # type: ignore[no-untyped-call]
            embedding_layer.embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.linear_layer = nn.Linear(
            (int(bidirectional) + 1) * num_layers * hidden_size, latent_size
        )

    @override
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = self.embedding_layer(x)
        packed_x = pack_padded_sequence(x, lengths, batch_first=True)
        lstm_out: LstmOutT = self.lstm_layer(packed_x)
        _, (_, x) = lstm_out
        x = x.permute(1, 2, 0).contiguous().view(batch_size, -1)
        return self.linear_layer(x)  # type: ignore[no-any-return]


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_layer: nn.Embedding,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        latent_size: int,
    ) -> None:
        super().__init__()

        self.latent2hidden_layer = nn.Linear(latent_size, hidden_size)
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
        states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        is_latent_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if is_latent_states:
            c0: torch.Tensor = self.latent2hidden_layer(states)
            c0 = c0.unsqueeze(0).repeat(self.lstm_layer.num_layers, 1, 1)
            h0 = torch.zeros_like(c0)
            states = (h0, c0)

        if not isinstance(states, tuple):
            raise TypeError("States must be a tuple or is_latent_states must be True")

        x = self.embedding_layer(x)
        packed_x = pack_padded_sequence(x, lengths, batch_first=True)
        lstm_out: LstmOutT = self.lstm_layer(packed_x, states)
        packed_x, _ = lstm_out
        x, lengths = pad_packed_sequence(packed_x, batch_first=True)
        x = self.linear_layer(x)

        return x, lengths, states


class Discriminator(nn.Module):
    def __init__(self, input_size: int, layers: Sequence[int]) -> None:
        super().__init__()

        layers = list(layers)
        in_features = [input_size, *layers]
        out_features = [*layers, 1]

        self.layers_seq = nn.Sequential()
        for k, (i, o) in enumerate(zip(in_features, out_features)):
            self.layers_seq.add_module("linear_{}".format(k), nn.Linear(i, o))
            if k != len(layers):
                self.layers_seq.add_module("activation_{}".format(k), nn.ELU(inplace=True))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers_seq(x)  # type: ignore[no-any-return]


class AAE(nn.Module):
    def __init__(self, vocabulary: CharVocab, config: AAEConfig) -> None:
        super().__init__()

        self.vocabulary = vocabulary
        self.latent_size = config.latent_size

        self.embeddings = nn.Embedding(
            len(vocabulary), config.embedding_size, padding_idx=vocabulary.pad
        )
        self.encoder = Encoder(
            self.embeddings,
            config.encoder_hidden_size,
            config.encoder_num_layers,
            config.encoder_bidirectional,
            config.encoder_dropout,
            config.latent_size,
        )
        self.decoder = Decoder(
            self.embeddings,
            config.decoder_hidden_size,
            config.decoder_num_layers,
            config.decoder_dropout,
            config.latent_size,
        )
        self.discriminator = Discriminator(config.latent_size, config.discriminator_layers)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device  # type: ignore[no-any-return]

    def encoder_forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, lengths)  # type: ignore[no-any-return]

    def decoder_forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        is_latent_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.decoder(x, lengths, states, is_latent_states)  # type: ignore[no-any-return]

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

    def sample_latent(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.latent_size, device=self.device)

    def sample(self, n_batch: int, max_len: int = 100) -> list[str]:
        with torch.no_grad():
            samples: list[torch.Tensor] = []
            lengths = torch.zeros(n_batch, dtype=torch.long, device=self.device)

            states = self.sample_latent(n_batch)
            prevs = torch.empty(n_batch, 1, dtype=torch.long, device=self.device).fill_(
                self.vocabulary.bos
            )
            one_lens = torch.ones(n_batch, dtype=torch.long, device="cpu")
            is_end = torch.zeros(n_batch, dtype=torch.uint8, device=self.device)

            for i in range(max_len):
                logits, _, states = self.decoder(prevs, one_lens, states, i == 0)
                logits = torch.softmax(logits, 2)
                shape = logits.shape[:-1]
                logits = logits.contiguous().view(-1, logits.shape[-1])
                currents = torch.distributions.Categorical(logits).sample()  # type:ignore[no-untyped-call]
                currents = currents.view(shape)

                is_end[currents.view(-1) == self.vocabulary.eos] = 1
                if is_end.sum() == max_len:
                    break

                currents[is_end, :] = self.vocabulary.pad
                samples.append(currents.cpu())
                lengths[~is_end] += 1

                prevs = currents

            if samples:
                cat_samples = torch.cat(samples, dim=-1)
                generated_samples = [
                    self.tensor2string(t[:l]) for t, l in zip(cat_samples, lengths)
                ]
            else:
                generated_samples = ["" for _ in range(n_batch)]

            return generated_samples
