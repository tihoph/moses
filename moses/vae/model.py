from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Sequence

    from moses.utils import GruOutT, OneHotVocab

    from .config import VAEConfig


class VAE(nn.Module):
    def __init__(self, vocab: OneHotVocab, config: VAEConfig) -> None:
        super().__init__()

        self.vocabulary = vocab
        # Special symbols
        self.pad = vocab.pad
        self.bos = vocab.bos

        # Word embeddings layer
        n_vocab, d_emb = len(vocab), vocab.vectors.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        self.x_emb.weight.data.copy_(vocab.vectors)
        if config.freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        # Encoder
        if config.q_cell == "gru":
            self.encoder_rnn = nn.GRU(  # type: ignore[no-untyped-call]
                d_emb,
                config.q_d_h,
                num_layers=config.q_n_layers,
                batch_first=True,
                dropout=config.q_dropout if config.q_n_layers > 1 else 0,
                bidirectional=config.q_bidir,
            )
        else:
            raise ValueError("Invalid q_cell type, should be one of the ('gru',)")

        q_d_last = config.q_d_h * (2 if config.q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, config.d_z)
        self.q_logvar = nn.Linear(q_d_last, config.d_z)

        # Decoder
        if config.d_cell == "gru":
            self.decoder_rnn = nn.GRU(  # type: ignore[no-untyped-call]
                d_emb + config.d_z,
                config.d_d_h,
                num_layers=config.d_n_layers,
                batch_first=True,
                dropout=config.d_dropout if config.d_n_layers > 1 else 0,
            )
        else:
            raise ValueError("Invalid d_cell type, should be one of the ('gru',)")

        self.decoder_lat = nn.Linear(config.d_z, config.d_d_h)
        self.decoder_fc = nn.Linear(config.d_d_h, n_vocab)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([self.encoder_rnn, self.q_mu, self.q_logvar])
        self.decoder = nn.ModuleList([self.decoder_rnn, self.decoder_lat, self.decoder_fc])
        self.vae = nn.ModuleList([self.x_emb, self.encoder, self.decoder])

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device  # type: ignore[no-any-return]

    def string2tensor(self, string: str, device: str | torch.device = "model") -> torch.Tensor:
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        return torch.tensor(
            ids, dtype=torch.long, device=self.device if device == "model" else device
        )

    def tensor2string(self, tensor: torch.Tensor) -> str:
        ids = tensor.tolist()
        return self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

    @override
    def forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder(x)

        # Decoder: x, z -> recon_loss
        recon_loss = self.forward_decoder(x, z)

        return kl_loss, recon_loss

    def forward_encoder(self, x: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        x = [self.x_emb(i_x) for i_x in x]
        packed_x = nn.utils.rnn.pack_sequence(x)

        _, h = self.encoder_rnn(packed_x, None)

        h = h[-(1 + int(self.encoder_rnn.bidirectional)) :]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps

        kl_loss = 0.5 * (logvar.exp() + mu**2 - 1 - logvar).sum(1).mean()

        return z, kl_loss

    def forward_decoder(self, x: Sequence[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """
        x = list(x)
        lengths = torch.Tensor([len(i_x) for i_x in x], device="cpu")

        padded_x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.pad)
        x_emb: torch.Tensor = self.x_emb(padded_x)

        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        packed_x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths, batch_first=True)

        h_0: torch.Tensor = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        rnn_output: GruOutT = self.decoder_rnn(packed_x_input, h_0)
        packed_output, _ = rnn_output
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        y: torch.Tensor = self.decoder_fc(output)

        return F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            padded_x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad,
        )

    def sample_z_prior(self, n_batch: int) -> torch.Tensor:
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.q_mu.out_features, device=self.x_emb.weight.device)

    def sample(
        self,
        n_batch: int,
        max_len: int = 100,
        z: torch.Tensor | None = None,
        temp: float = 1.0,
    ) -> list[str]:
        """Generating n_batch samples in eval mode (`z` could be
        not on same device)

        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :return: list of tensors of strings, samples sequence x
        """
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.device)
            z_0 = z.unsqueeze(1)

            # Initial values
            h: torch.Tensor = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch, max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=self.device).repeat(n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.uint8, device=self.device)

            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder_rnn(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)

                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = [x[i, : end_pads[i]] for i in range(x.size(0))]

            return [self.tensor2string(i_x) for i_x in new_x]
