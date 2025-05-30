from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from rdkit import Chem
from torch import autograd, nn
from torch.utils import data
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ddc_pub import ddc_v3 as ddc
    from numpy.typing import NDArray

    from moses.utils import CharVocab

    from .config import LatentGANConfig


class LatentGAN(nn.Module):
    def __init__(self, vocabulary: CharVocab, config: LatentGANConfig) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.Generator = Generator(data_shape=(1, config.latent_vector_dim))
        self.model_version = config.heteroencoder_version
        self.Discriminator = Discriminator(data_shape=(1, config.latent_vector_dim))
        self.sample_decoder: ddc.DDC | None = None
        self.model_loaded = False
        self.new_batch_size = 256
        # init params
        cuda = torch.cuda.is_available()
        if cuda:
            self.Discriminator.cuda()
            self.Generator.cuda()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # type:ignore[attr-defined]

    @override
    def forward(self, n_batch: int) -> NDArray[np.str_]:
        return self.sample(n_batch)

    def encode_smiles(
        self,
        smiles_in: Sequence[str],
        encoder: Literal["chembl", "moses", "new"] | None = None,
    ) -> list[list[int]]:
        model = load_model(model_version=encoder)

        # MUST convert SMILES to binary mols for the model to accept them
        # (it re-converts them to SMILES internally)
        mols_in = [Chem.rdchem.Mol.ToBinary(Chem.MolFromSmiles(smiles)) for smiles in smiles_in]
        latent = model.transform(model.vectorize(mols_in))

        return latent.tolist()  # type: ignore[no-any-return]

    def compute_gradient_penalty(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        discriminator: Discriminator,
    ) -> torch.Tensor:
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_samples.size(0), 1)))

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = discriminator(interpolates)
        fake = self.Tensor(real_samples.shape[0], 1).fill_(1.0)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # type: ignore[no-any-return]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device  # type: ignore[no-any-return]

    def sample(self, n_batch: int, max_length: int = 100) -> NDArray[np.str_]:
        if not self.model_loaded or not self.sample_decoder:
            # Checking for first batch of model to only load model once
            print("Heteroencoder for Sampling Loaded")
            self.sample_decoder = load_model(model_version=self.model_version)
            # load generator

            self.Gen = self.Generator
            self.Gen.eval()

            self.D = self.Discriminator
            torch.no_grad()
            cuda = torch.cuda.is_available()
            if cuda:
                self.Gen.cuda()
                self.D.cuda()
            self.S = Sampler(generator=self.Gen)
            self.model_loaded = True

            if n_batch <= 256:
                print(
                    "Batch size of {} detected. Decoding "
                    "performs poorly when Batch size != 256. \
                 Setting batch size to 256".format(n_batch)
                )
        # Sampling performs very poorly on default sampling batch parameters.
        #  This takes care of the default scenario.
        if n_batch == 32:
            n_batch = 256

        latent = self.S.sample(n_batch)
        lat = latent.detach().cpu().numpy()

        if self.new_batch_size != n_batch:
            # The batch decoder creates a new instance of the decoder
            # every time a new batch size is given, e.g. for the
            # final batch of the generation.
            self.new_batch_size = n_batch
            self.sample_decoder.batch_input_length = self.new_batch_size

        sys.stdout.flush()

        smi, _ = self.sample_decoder.predict_batch(lat, temp=0)
        return smi  # type: ignore[no-any-return]


def load_model(
    model_version: Literal["chembl", "moses", "new"] | None = None,
) -> ddc.DDC:
    from ddc_pub import ddc_v3 as ddc

    # Import model
    currentDirectory = os.getcwd()

    if model_version == "chembl":
        model_name = "chembl_pretrained"
    elif model_version == "moses":
        model_name = "moses_pretrained"
    elif model_version == "new":
        model_name = "new_model"
    else:
        print(
            "No predefined model of that name found. "
            "using the default pre-trained MOSES heteroencoder"
        )
        model_name = "moses_pretrained"

    path = "{}/moses/latentgan/heteroencoder_models/{}".format(currentDirectory, model_name)
    print("Loading heteroencoder model titled {}".format(model_version))
    print("Path to model file: {}".format(path))
    model = ddc.DDC(model_name=path)
    sys.stdout.flush()

    return model


class LatentMolsDataset(data.Dataset[str]):
    def __init__(self, latent_space_mols: Sequence[str]) -> None:
        self.data = latent_space_mols

    def __len__(self) -> int:
        return len(self.data)

    @override
    def __getitem__(self, index: int) -> str:
        return self.data[index]


class Discriminator(nn.Module):
    def __init__(self, data_shape: tuple[int, int] = (1, 512)) -> None:
        super().__init__()
        self.data_shape = data_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.data_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    @override
    def forward(self, mol: torch.Tensor) -> torch.Tensor:
        return self.model(mol)  # type: ignore[no-any-return]


class Generator(nn.Module):
    def __init__(
        self, data_shape: tuple[int, int] = (1, 512), latent_dim: int | None = None
    ) -> None:
        super().__init__()
        self.data_shape = data_shape

        # latent dim of the generator is one of the hyperparams.
        # by default it is set to the prod of data_shapes
        self.latent_dim = int(np.prod(self.data_shape)) if latent_dim is None else latent_dim

        def block(in_feat: int, out_feat: int, normalize: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.data_shape))),
            # nn.Tanh() # expecting latent vectors to be not normalized # noqa: ERA001
        )

    @override
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.model(z)
        return out


class Sampler:
    """
    Sampling the mols the generator.
    All scripts should use this class for sampling.
    """

    def __init__(self, generator: Generator) -> None:
        self.G = generator

    def sample(self, n: int) -> torch.Tensor:
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.uniform(-1, 1, (n, self.G.latent_dim)))  # type: ignore[attr-defined]
        # Generate a batch of mols
        return self.G(z)  # type: ignore[no-any-return]
