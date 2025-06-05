from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch import nn
from typing_extensions import override

if TYPE_CHECKING:
    from moses.utils import CharVocab, LstmOutT

    from .config import CharRNNConfig


class CharRNN(nn.Module):
    def __init__(self, vocabulary: CharVocab, config: CharRNNConfig) -> None:
        super().__init__()

        self.vocabulary = vocabulary
        self.hidden_size = config.hidden
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.vocab_size = self.input_size = self.output_size = len(vocabulary)

        self.embedding_layer = nn.Embedding(
            self.vocab_size, self.vocab_size, padding_idx=vocabulary.pad
        )
        self.lstm_layer = nn.LSTM(  # type: ignore[no-untyped-call]
            self.input_size,
            self.hidden_size,
            self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device  # type: ignore[no-any-return]

    @override
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        hiddens: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding_layer(x)
        packed_x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True)
        lstm_out: LstmOutT = self.lstm_layer(packed_x, hiddens)
        packed_x, hiddens = lstm_out
        x, _ = rnn_utils.pad_packed_sequence(packed_x, batch_first=True)
        x = self.linear_layer(x)

        return x, lengths, hiddens

    def string2tensor(self, string: str, device: str | torch.device = "model") -> torch.Tensor:
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        return torch.tensor(
            ids, dtype=torch.long, device=self.device if device == "model" else device
        )

    def tensor2string(self, tensor: torch.Tensor) -> str:
        ids = tensor.tolist()
        return self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

    def sample(self, n_batch: int, max_length: int = 100) -> list[str]:
        with torch.no_grad():
            starts_ls = [
                torch.tensor([self.vocabulary.bos], dtype=torch.long, device=self.device)
                for _ in range(n_batch)
            ]

            starts = torch.tensor(starts_ls, dtype=torch.long, device=self.device).unsqueeze(1)

            new_smiles_list = [
                torch.tensor(self.vocabulary.pad, dtype=torch.long, device=self.device).repeat(
                    max_length + 2
                )
                for _ in range(n_batch)
            ]

            for i in range(n_batch):
                new_smiles_list[i][0] = self.vocabulary.bos

            len_smiles_list = [1 for _ in range(n_batch)]
            lens = torch.tensor([1 for _ in range(n_batch)], dtype=torch.long, device="cpu")
            end_smiles_list = [False for _ in range(n_batch)]

            hiddens = None
            for i in range(1, max_length + 1):
                output, _, hiddens = self.forward(starts, lens, hiddens)

                # probabilities
                probs = [F.softmax(o, dim=-1) for o in output]

                # sample from probabilities
                ind_tops = [torch.multinomial(p, 1) for p in probs]

                for j, top in enumerate(ind_tops):
                    if not end_smiles_list[j]:
                        top_elem = top[0].item()
                        if top_elem == self.vocabulary.eos:
                            end_smiles_list[j] = True

                        new_smiles_list[j][i] = top_elem
                        len_smiles_list[j] = len_smiles_list[j] + 1

                starts = torch.tensor(ind_tops, dtype=torch.long, device=self.device).unsqueeze(1)

            new_smiles_list = [new_smiles_list[i][:l] for i, l in enumerate(len_smiles_list)]
            return [self.tensor2string(t) for t in new_smiles_list]
