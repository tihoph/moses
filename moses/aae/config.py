from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class AAEConfig(argparse.Namespace):
    embedding_size: int = 32
    encoder_hidden_size: int = 512
    encoder_num_layers: int = 1
    encoder_bidirectional: bool = True
    encoder_dropout: float = 0.0
    decoder_hidden_size: int = 512
    decoder_num_layers: int = 2
    decoder_dropout: float = 0.0
    latent_size: int = 128
    discriminator_layers: Sequence[int] = (640, 256)
    pretrain_epochs: int = 0
    train_epochs: int = 120
    n_batch: int = 512
    lr: float = 1e-3
    step_size: int = 20
    gamma: float = 0.5
    n_jobs: int = 1
    n_workers: int = 1
    discriminator_steps: int = 1
    weight_decay: int = 0


def get_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()

    model_arg = parser.add_argument_group("Model")
    model_arg.add_argument(
        "--embedding_size",
        type=int,
        default=AAEConfig.embedding_size,
        help="Embedding size in encoder and decoder",
    )
    model_arg.add_argument(
        "--encoder_hidden_size",
        type=int,
        default=AAEConfig.encoder_hidden_size,
        help="Size of hidden state for lstm layers in encoder",
    )
    model_arg.add_argument(
        "--encoder_num_layers",
        type=int,
        default=AAEConfig.encoder_num_layers,
        help="Number of lstm layers in encoder",
    )
    model_arg.add_argument(
        "--encoder_bidirectional",
        type=bool,
        default=AAEConfig.encoder_bidirectional,
        help="If true to use bidirectional lstm layers in encoder",
    )
    model_arg.add_argument(
        "--encoder_dropout",
        type=float,
        default=AAEConfig.encoder_dropout,
        help="Dropout probability for lstm layers in encoder",
    )
    model_arg.add_argument(
        "--decoder_hidden_size",
        type=int,
        default=AAEConfig.decoder_hidden_size,
        help="Size of hidden state for lstm layers in decoder",
    )
    model_arg.add_argument(
        "--decoder_num_layers",
        type=int,
        default=AAEConfig.decoder_num_layers,
        help="Number of lstm layers in decoder",
    )
    model_arg.add_argument(
        "--decoder_dropout",
        type=float,
        default=AAEConfig.decoder_dropout,
        help="Dropout probability for lstm layers in decoder",
    )
    model_arg.add_argument(
        "--latent_size", type=int, default=AAEConfig.latent_size, help="Size of latent vectors"
    )
    model_arg.add_argument(
        "--discriminator_layers",
        nargs="+",
        type=int,
        default=AAEConfig.discriminator_layers,
        help="Numbers of features for linear layers in discriminator",
    )

    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument(
        "--pretrain_epochs",
        type=int,
        default=AAEConfig.pretrain_epochs,
        help="Number of epochs for autoencoder pretraining",
    )
    train_arg.add_argument(
        "--train_epochs",
        type=int,
        default=AAEConfig.train_epochs,
        help="Number of epochs for autoencoder training",
    )
    train_arg.add_argument("--n_batch", type=int, default=AAEConfig.n_batch, help="Size of batch")
    train_arg.add_argument("--lr", type=float, default=AAEConfig.lr, help="Learning rate")
    train_arg.add_argument(
        "--step_size", type=float, default=AAEConfig.step_size, help="Period of learning rate decay"
    )
    train_arg.add_argument(
        "--gamma",
        type=float,
        default=AAEConfig.gamma,
        help="Multiplicative factor of learning rate decay",
    )
    train_arg.add_argument("--n_jobs", type=int, default=AAEConfig.n_jobs, help="Number of threads")
    train_arg.add_argument(
        "--n_workers",
        type=int,
        default=AAEConfig.n_workers,
        help="Number of workers for DataLoaders",
    )
    train_arg.add_argument(
        "--discriminator_steps",
        type=int,
        default=AAEConfig.discriminator_steps,
        help="Discriminator training steps per oneautoencoder training step",
    )
    train_arg.add_argument(
        "--weight_decay",
        type=int,
        default=AAEConfig.weight_decay,
        help="weight decay for optimizer",
    )

    return parser


def get_config() -> AAEConfig:
    parser = get_parser()
    return parser.parse_known_args(namespace=AAEConfig())[0]
