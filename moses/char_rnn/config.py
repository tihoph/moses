from __future__ import annotations

import argparse


class CharRNNConfig(argparse.Namespace):
    num_layers: int = 3
    hidden: int = 768
    dropout: float = 0.2
    train_epochs: int = 80
    n_batch: int = 64
    lr: float = 1e-3
    step_size: int = 10
    gamma: float = 0.5
    n_jobs: int = 1
    n_workers: int = 1


def get_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group("Model")
    model_arg.add_argument(
        "--num_layers", type=int, default=CharRNNConfig.num_layers, help="Number of LSTM layers"
    )
    model_arg.add_argument("--hidden", type=int, default=CharRNNConfig.hidden, help="Hidden size")
    model_arg.add_argument(
        "--dropout",
        type=float,
        default=CharRNNConfig.dropout,
        help="dropout between LSTM layers except for last",
    )

    # Train
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument(
        "--train_epochs",
        type=int,
        default=CharRNNConfig.train_epochs,
        help="Number of epochs for model training",
    )
    train_arg.add_argument(
        "--n_batch", type=int, default=CharRNNConfig.n_batch, help="Size of batch"
    )
    train_arg.add_argument("--lr", type=float, default=CharRNNConfig.lr, help="Learning rate")
    train_arg.add_argument(
        "--step_size",
        type=int,
        default=CharRNNConfig.step_size,
        help="Period of learning rate decay",
    )
    train_arg.add_argument(
        "--gamma",
        type=float,
        default=CharRNNConfig.gamma,
        help="Multiplicative factor of learning rate decay",
    )
    train_arg.add_argument(
        "--n_jobs", type=int, default=CharRNNConfig.n_jobs, help="Number of threads"
    )
    train_arg.add_argument(
        "--n_workers",
        type=int,
        default=CharRNNConfig.n_workers,
        help="Number of workers for DataLoaders",
    )

    return parser


def get_config() -> CharRNNConfig:
    parser = get_parser()
    return parser.parse_known_args(namespace=CharRNNConfig())[0]
