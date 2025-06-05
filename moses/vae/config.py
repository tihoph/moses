from __future__ import annotations

import argparse
from typing import Literal


class VAEConfig(argparse.Namespace):
    q_cell: Literal["gru"] = "gru"
    q_bidir: bool = False
    q_d_h: int = 256
    q_n_layers: int = 1
    q_dropout: float = 0.5
    d_cell: Literal["gru"] = "gru"
    d_n_layers: int = 3
    d_dropout: float = 0
    d_z: int = 128
    d_d_h: int = 512
    freeze_embeddings: bool = False
    n_batch: int = 512
    clip_grad: int = 50
    kl_start: int = 0
    kl_w_start: float = 0
    kl_w_end: float = 0.05
    lr_start: float = 3 * 1e-4
    lr_n_period: int = 10
    lr_n_restarts: int = 10
    lr_n_mult: int = 1
    lr_end: float = 3 * 1e-4
    n_last: int = 1000
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
        "--q_cell",
        type=str,
        default=VAEConfig.q_cell,
        choices=["gru"],
        help="Encoder rnn cell type",
    )
    model_arg.add_argument(
        "--q_bidir",
        default=VAEConfig.q_bidir,
        action="store_true",
        help="If to add second direction to encoder",
    )
    model_arg.add_argument(
        "--q_d_h", type=int, default=VAEConfig.q_d_h, help="Encoder h dimensionality"
    )
    model_arg.add_argument(
        "--q_n_layers", type=int, default=VAEConfig.q_n_layers, help="Encoder number of layers"
    )
    model_arg.add_argument(
        "--q_dropout", type=float, default=VAEConfig.q_dropout, help="Encoder layers dropout"
    )
    model_arg.add_argument(
        "--d_cell",
        type=str,
        default=VAEConfig.d_cell,
        choices=["gru"],
        help="Decoder rnn cell type",
    )
    model_arg.add_argument(
        "--d_n_layers", type=int, default=VAEConfig.d_n_layers, help="Decoder number of layers"
    )
    model_arg.add_argument(
        "--d_dropout", type=float, default=VAEConfig.d_dropout, help="Decoder layers dropout"
    )
    model_arg.add_argument(
        "--d_z", type=int, default=VAEConfig.d_z, help="Latent vector dimensionality"
    )
    model_arg.add_argument(
        "--d_d_h", type=int, default=VAEConfig.d_d_h, help="Decoder hidden dimensionality"
    )
    model_arg.add_argument(
        "--freeze_embeddings",
        default=VAEConfig.freeze_embeddings,
        action="store_true",
        help="If to freeze embeddings while training",
    )

    # Train
    train_arg = parser.add_argument_group("Train")
    train_arg.add_argument("--n_batch", type=int, default=VAEConfig.n_batch, help="Batch size")
    train_arg.add_argument(
        "--clip_grad", type=int, default=VAEConfig.clip_grad, help="Clip gradients to this value"
    )
    train_arg.add_argument(
        "--kl_start",
        type=int,
        default=VAEConfig.kl_start,
        help="Epoch to start change kl weight from",
    )
    train_arg.add_argument(
        "--kl_w_start", type=float, default=VAEConfig.kl_w_start, help="Initial kl weight value"
    )
    train_arg.add_argument(
        "--kl_w_end", type=float, default=VAEConfig.kl_w_end, help="Maximum kl weight value"
    )
    train_arg.add_argument(
        "--lr_start", type=float, default=VAEConfig.lr_start, help="Initial lr value"
    )
    train_arg.add_argument(
        "--lr_n_period",
        type=int,
        default=VAEConfig.lr_n_period,
        help="Epochs before first restart in SGDR",
    )
    train_arg.add_argument(
        "--lr_n_restarts",
        type=int,
        default=VAEConfig.lr_n_restarts,
        help="Number of restarts in SGDR",
    )
    train_arg.add_argument(
        "--lr_n_mult",
        type=int,
        default=VAEConfig.lr_n_mult,
        help="Mult coefficient after restart in SGDR",
    )
    train_arg.add_argument(
        "--lr_end", type=float, default=VAEConfig.lr_end, help="Maximum lr weight value"
    )
    train_arg.add_argument(
        "--n_last", type=int, default=VAEConfig.n_last, help="Number of iters to smooth loss calc"
    )
    train_arg.add_argument("--n_jobs", type=int, default=VAEConfig.n_jobs, help="Number of threads")
    train_arg.add_argument(
        "--n_workers",
        type=int,
        default=VAEConfig.n_workers,
        help="Number of workers for DataLoaders",
    )

    return parser
