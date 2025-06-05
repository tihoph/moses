from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

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
    embedding_size: int = 32
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.0
    discriminator_layers: Sequence[tuple[int, int]] = (
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
    )
    discriminator_dropout: float = 0.0
    reward_weight: float = 0.7
    generator_pretrain_epochs: int = 50
    discriminator_pretrain_epochs: int = 50
    pg_iters: int = 1000
    n_batch: int = 64
    lr: float = 1e-4
    n_jobs: int = 8
    n_workers: int = 1
    max_length: int = 100
    clip_grad: float = 5
    rollouts: int = 16
    generator_updates: int = 1
    discriminator_updates: int = 1
    discriminator_epochs: int = 10
    pg_smooth_const: float = 0.1
    n_ref_subsample: int = 500
    additional_rewards: Collection[MetricT] = SUPPORTED_METRICS


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
        default=ORGANConfig.embedding_size,
        help="Embedding size in generator and discriminator",
    )
    model_arg.add_argument(
        "--hidden_size",
        type=int,
        default=ORGANConfig.hidden_size,
        help="Size of hidden state for lstm layers in generator",
    )
    model_arg.add_argument(
        "--num_layers",
        type=int,
        default=ORGANConfig.num_layers,
        help="Number of lstm layers in generator",
    )
    model_arg.add_argument(
        "--dropout",
        type=float,
        default=ORGANConfig.dropout,
        help="Dropout probability for lstm layers in generator",
    )
    model_arg.add_argument(
        "--discriminator_layers",
        nargs="+",
        type=conv_pair,
        default=ORGANConfig.discriminator_layers,
        help="Numbers of features for convalution layers in discriminator",
    )
    model_arg.add_argument(
        "--discriminator_dropout",
        type=float,
        default=ORGANConfig.discriminator_dropout,
        help="Dropout probability for discriminator",
    )
    model_arg.add_argument(
        "--reward_weight",
        type=restricted_float,
        default=ORGANConfig.reward_weight,
        help="Reward weight for policy gradient training",
    )

    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument(
        "--generator_pretrain_epochs",
        type=int,
        default=ORGANConfig.generator_pretrain_epochs,
        help="Number of epochs for generator pretraining",
    )
    train_arg.add_argument(
        "--discriminator_pretrain_epochs",
        type=int,
        default=ORGANConfig.discriminator_pretrain_epochs,
        help="Number of epochs for discriminator pretraining",
    )
    train_arg.add_argument(
        "--pg_iters",
        type=int,
        default=ORGANConfig.pg_iters,
        help="Number of iterations for policy gradient training",
    )
    train_arg.add_argument("--n_batch", type=int, default=ORGANConfig.n_batch, help="Size of batch")
    train_arg.add_argument("--lr", type=float, default=ORGANConfig.lr, help="Learning rate")
    train_arg.add_argument(
        "--n_jobs", type=int, default=ORGANConfig.n_jobs, help="Number of threads"
    )

    train_arg.add_argument(
        "--n_workers",
        type=int,
        default=ORGANConfig.n_workers,
        help="Number of workers for DataLoaders",
    )
    train_arg.add_argument(
        "--max_length", type=int, default=ORGANConfig.max_length, help="Maximum length for sequence"
    )
    train_arg.add_argument(
        "--clip_grad",
        type=float,
        default=ORGANConfig.max_length,
        help="Clip PG generator gradients to this value",
    )
    train_arg.add_argument(
        "--rollouts", type=int, default=ORGANConfig.rollouts, help="Number of rollouts"
    )
    train_arg.add_argument(
        "--generator_updates",
        type=int,
        default=ORGANConfig.generator_updates,
        help="Number of updates of generator per iteration",
    )
    train_arg.add_argument(
        "--discriminator_updates",
        type=int,
        default=ORGANConfig.discriminator_updates,
        help="Number of updates of discriminator per iteration",
    )
    train_arg.add_argument(
        "--discriminator_epochs",
        type=int,
        default=ORGANConfig.discriminator_epochs,
        help="Number of epochs of discriminator per iteration",
    )
    train_arg.add_argument(
        "--pg_smooth_const",
        type=float,
        default=ORGANConfig.pg_smooth_const,
        help="Smoothing factor for Policy Gradient logs",
    )
    parser.add_argument(
        "--n_ref_subsample",
        type=int,
        default=ORGANConfig.pg_smooth_const,
        help="Number of reference molecules (sampling from training data)",
    )
    parser.add_argument(
        "--additional_rewards",
        nargs="+",
        type=str,
        choices=ORGANConfig.additional_rewards,
        default=[],
        help="Adding of addition rewards",
    )
    return parser


def get_config() -> ORGANConfig:
    parser = get_parser()
    return parser.parse_known_args(namespace=ORGANConfig())[0]
