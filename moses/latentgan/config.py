from __future__ import annotations

import argparse
from typing import Literal


class LatentGANConfig(argparse.Namespace):
    heteroencoder_version: Literal["chembl", "moses", "new"] = "moses"
    gp: int = 10
    n_critic: int = 5
    train_epochs: int = 2000
    n_batch: int = 64
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999
    step_size: int = 10
    latent_vector_dim: int = 512
    gamma: float = 1.0
    n_jobs: int = 1
    n_workers: int = 1
    heteroencoder_layer_dim: int = 512
    heteroencoder_noise_std: float = 0.1
    heteroencoder_dec_layers: int = 4
    heteroencoder_batch_size: int = 128
    heteroencoder_epochs: int = 100
    heteroencoder_lr: float = 1e-3
    heteroencoder_mini_epochs: int = 10
    heteroencoder_lr_decay: bool = True
    heteroencoder_patience: int = 100
    heteroencoder_lr_decay_start: int = 500
    heteroencoder_save_period: int = 100


def get_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group("Model")
    model_arg.add_argument(
        "--heteroencoder_version",
        choices=["chembl", "moses", "new"],
        default=LatentGANConfig.heteroencoder_version,
        help="Which heteroencoder model version to use",
    )
    # Train
    train_arg = parser.add_argument_group("Training")

    train_arg.add_argument(
        "--gp", type=int, default=LatentGANConfig.gp, help="Gradient Penalty Coefficient"
    )
    train_arg.add_argument(
        "--n_critic",
        type=int,
        default=LatentGANConfig.n_critic,
        help="Ratio of discriminator to generator training frequency",
    )
    train_arg.add_argument(
        "--train_epochs",
        type=int,
        default=LatentGANConfig.train_epochs,
        help="Number of epochs for model training",
    )
    train_arg.add_argument(
        "--n_batch", type=int, default=LatentGANConfig.n_batch, help="Size of batch"
    )
    train_arg.add_argument("--lr", type=float, default=LatentGANConfig.lr, help="Learning rate")
    train_arg.add_argument(
        "--b1", type=float, default=LatentGANConfig.b1, help="Adam optimizer parameter beta 1"
    )
    train_arg.add_argument(
        "--b2", type=float, default=LatentGANConfig.b2, help="Adam optimizer parameter beta 2"
    )
    train_arg.add_argument(
        "--step_size",
        type=int,
        default=LatentGANConfig.step_size,
        help="Period of learning rate decay",
    )
    train_arg.add_argument(
        "--latent_vector_dim",
        type=int,
        default=LatentGANConfig.latent_vector_dim,
        help="Size of latentgan vector",
    )
    train_arg.add_argument(
        "--gamma",
        type=float,
        default=LatentGANConfig.gamma,
        help="Multiplicative factor of learning rate decay",
    )
    train_arg.add_argument(
        "--n_jobs", type=int, default=LatentGANConfig.n_jobs, help="Number of threads"
    )
    train_arg.add_argument(
        "--n_workers",
        type=int,
        default=LatentGANConfig.n_workers,
        help="Number of workers for DataLoaders",
    )

    # Arguments used if training a new heteroencoder
    heteroencoder_arg = parser.add_argument_group("heteroencoder")

    heteroencoder_arg.add_argument(
        "--heteroencoder_layer_dim",
        type=int,
        default=LatentGANConfig.heteroencoder_layer_dim,
        help="Layer size for heteroencoder (if training new heteroencoder)",
    )
    heteroencoder_arg.add_argument(
        "--heteroencoder_noise_std",
        type=float,
        default=LatentGANConfig.heteroencoder_noise_std,
        help="Noise amplitude for heteroencoder",
    )
    heteroencoder_arg.add_argument(
        "--heteroencoder_dec_layers",
        type=int,
        default=LatentGANConfig.heteroencoder_dec_layers,
        help="Number of decoding layers for heteroencoder",
    )
    heteroencoder_arg.add_argument(
        "--heteroencoder_batch_size",
        type=int,
        default=LatentGANConfig.heteroencoder_batch_size,
        help="Batch size for heteroencoder",
    )
    heteroencoder_arg.add_argument(
        "--heteroencoder_epochs",
        type=int,
        default=LatentGANConfig.heteroencoder_epochs,
        help="Number of epochs for heteroencoder",
    )
    heteroencoder_arg.add_argument(
        "--heteroencoder_lr",
        type=float,
        default=LatentGANConfig.heteroencoder_lr,
        help="learning rate for heteroencoder",
    )
    heteroencoder_arg.add_argument(
        "--heteroencoder_mini_epochs",
        type=int,
        default=LatentGANConfig.heteroencoder_mini_epochs,
        help="How many sub-epochs to split each epoch for heteroencoder",
    )
    heteroencoder_arg.add_argument(
        "--heteroencoder_lr_decay",
        default=LatentGANConfig.heteroencoder_lr_decay,
        action="store_false",
        help="Use learning rate decay for heteroencoder ",
    )
    heteroencoder_arg.add_argument(
        "--heteroencoder_patience",
        type=int,
        default=LatentGANConfig.heteroencoder_patience,
        help="Patience for adaptive learning rate for heteroencoder",
    )
    heteroencoder_arg.add_argument(
        "--heteroencoder_lr_decay_start",
        type=int,
        default=LatentGANConfig.heteroencoder_lr_decay_start,
        help="Which sub-epoch to start decaying learning rate for heteroencoder ",
    )
    heteroencoder_arg.add_argument(
        "--heteroencoder_save_period",
        type=int,
        default=LatentGANConfig.heteroencoder_save_period,
        help="How often in sub-epochs to save model checkpoints for heteroencoder",
    )

    return parser


def get_config() -> LatentGANConfig:
    parser = get_parser()
    return parser.parse_known_args(namespace=LatentGANConfig())[0]
