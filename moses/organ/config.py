from __future__ import annotations

import argparse
from typing import Literal, get_args

MetricT = Literal[
    "fcd",
    "snn",
    "fragments",
    "scaffolds",
    "internal_diversity",
    "filters",
    "logp",
    "sa",
    "qed",
    "weight",
]
SUPPORTED_METRICS: frozenset[MetricT] = frozenset(get_args(MetricT))


class ORGANConfig(argparse.Namespace):
    embedding_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    discriminator_layers: list[tuple[int, int]]
    discriminator_dropout: float
    reward_weight: float
    generator_pretrain_epochs: int
    discriminator_pretrain_epochs: int
    pg_iters: int
    n_batch: int
    lr: float
    n_jobs: int
    n_workers: int
    max_length: int
    clip_grad: float
    rollouts: int
    generator_updates: int
    discriminator_updates: int
    discriminator_epochs: int
    pg_smooth_const: float
    n_ref_subsample: int
    additional_rewards: list[MetricT]


def get_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    def restricted_float(arg: str) -> float:
        if float(arg) < 0 or float(arg) > 1:
            raise argparse.ArgumentTypeError("{} not in range [0, 1]".format(arg))
        return float(arg)

    def conv_pair(arg: str) -> tuple[int, int]:
        if arg[0] != "(" or arg[-1] != ")":
            raise argparse.ArgumentTypeError("Wrong pair: {}".format(arg))

        feats, kernel_size = arg[1:-1].split(",")
        feats_int, kernel_size_int = int(feats), int(kernel_size)

        return feats_int, kernel_size_int

    if parser is None:
        parser = argparse.ArgumentParser()

    model_arg = parser.add_argument_group("Model")
    model_arg.add_argument(
        "--embedding_size",
        type=int,
        default=32,
        help="Embedding size in generator and discriminator",
    )
    model_arg.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="Size of hidden state for lstm layers in generator",
    )
    model_arg.add_argument(
        "--num_layers", type=int, default=2, help="Number of lstm layers in generator"
    )
    model_arg.add_argument(
        "--dropout",
        type=float,
        default=0,
        help="Dropout probability for lstm layers in generator",
    )
    model_arg.add_argument(
        "--discriminator_layers",
        nargs="+",
        type=conv_pair,
        default=[
            (100, 1),
            (200, 2),
            (200, 3),
            (200, 4),
            (200, 5),
            (100, 6),
            (100, 7),
            (100, 8),
            (100, 9),
            (100, 10),
            (160, 15),
            (160, 20),
        ],
        help="Numbers of features for convalution layers in discriminator",
    )
    model_arg.add_argument(
        "--discriminator_dropout",
        type=float,
        default=0,
        help="Dropout probability for discriminator",
    )
    model_arg.add_argument(
        "--reward_weight",
        type=restricted_float,
        default=0.7,
        help="Reward weight for policy gradient training",
    )

    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument(
        "--generator_pretrain_epochs",
        type=int,
        default=50,
        help="Number of epochs for generator pretraining",
    )
    train_arg.add_argument(
        "--discriminator_pretrain_epochs",
        type=int,
        default=50,
        help="Number of epochs for discriminator pretraining",
    )
    train_arg.add_argument(
        "--pg_iters",
        type=int,
        default=1000,
        help="Number of iterations for policy gradient training",
    )
    train_arg.add_argument("--n_batch", type=int, default=64, help="Size of batch")
    train_arg.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_arg.add_argument("--n_jobs", type=int, default=8, help="Number of threads")

    train_arg.add_argument(
        "--n_workers", type=int, default=1, help="Number of workers for DataLoaders"
    )
    train_arg.add_argument(
        "--max_length", type=int, default=100, help="Maximum length for sequence"
    )
    train_arg.add_argument(
        "--clip_grad",
        type=float,
        default=5,
        help="Clip PG generator gradients to this value",
    )
    train_arg.add_argument("--rollouts", type=int, default=16, help="Number of rollouts")
    train_arg.add_argument(
        "--generator_updates",
        type=int,
        default=1,
        help="Number of updates of generator per iteration",
    )
    train_arg.add_argument(
        "--discriminator_updates",
        type=int,
        default=1,
        help="Number of updates of discriminator per iteration",
    )
    train_arg.add_argument(
        "--discriminator_epochs",
        type=int,
        default=10,
        help="Number of epochs of discriminator per iteration",
    )
    train_arg.add_argument(
        "--pg_smooth_const",
        type=float,
        default=0.1,
        help="Smoothing factor for Policy Gradient logs",
    )

    parser.add_argument(
        "--n_ref_subsample",
        type=int,
        default=500,
        help="Number of reference molecules (sampling from training data)",
    )
    parser.add_argument(
        "--additional_rewards",
        nargs="+",
        type=str,
        choices=SUPPORTED_METRICS,
        default=[],
        help="Adding of addition rewards",
    )
    return parser


def get_config() -> ORGANConfig:
    parser = get_parser()
    return parser.parse_known_args()[0]  # type:ignore[return-value]
