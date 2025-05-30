from __future__ import annotations

import argparse
from typing import Literal


class VAEConfig(argparse.Namespace):
    q_cell: Literal["gru"]
    q_bidir: bool
    q_d_h: int
    q_n_layers: int
    q_dropout: float
    d_cell: Literal["gru"]
    d_n_layers: int
    d_dropout: float
    d_z: int
    d_d_h: int
    freeze_embeddings: bool
    n_batch: int
    clip_grad: int
    kl_start: int
    kl_w_start: int
    kl_w_end: int
    lr_start: float
    lr_n_period: int
    lr_n_restarts: int
    lr_n_mult: int
    lr_end: float
    n_last: int
    n_jobs: int
    n_workers: int


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
        default="gru",
        choices=["gru"],
        help="Encoder rnn cell type",
    )
    model_arg.add_argument(
        "--q_bidir",
        default=False,
        action="store_true",
        help="If to add second direction to encoder",
    )
    model_arg.add_argument("--q_d_h", type=int, default=256, help="Encoder h dimensionality")
    model_arg.add_argument("--q_n_layers", type=int, default=1, help="Encoder number of layers")
    model_arg.add_argument("--q_dropout", type=float, default=0.5, help="Encoder layers dropout")
    model_arg.add_argument(
        "--d_cell",
        type=str,
        default="gru",
        choices=["gru"],
        help="Decoder rnn cell type",
    )
    model_arg.add_argument("--d_n_layers", type=int, default=3, help="Decoder number of layers")
    model_arg.add_argument("--d_dropout", type=float, default=0, help="Decoder layers dropout")
    model_arg.add_argument("--d_z", type=int, default=128, help="Latent vector dimensionality")
    model_arg.add_argument("--d_d_h", type=int, default=512, help="Decoder hidden dimensionality")
    model_arg.add_argument(
        "--freeze_embeddings",
        default=False,
        action="store_true",
        help="If to freeze embeddings while training",
    )

    # Train
    train_arg = parser.add_argument_group("Train")
    train_arg.add_argument("--n_batch", type=int, default=512, help="Batch size")
    train_arg.add_argument("--clip_grad", type=int, default=50, help="Clip gradients to this value")
    train_arg.add_argument(
        "--kl_start", type=int, default=0, help="Epoch to start change kl weight from"
    )
    train_arg.add_argument("--kl_w_start", type=float, default=0, help="Initial kl weight value")
    train_arg.add_argument("--kl_w_end", type=float, default=0.05, help="Maximum kl weight value")
    train_arg.add_argument("--lr_start", type=float, default=3 * 1e-4, help="Initial lr value")
    train_arg.add_argument(
        "--lr_n_period",
        type=int,
        default=10,
        help="Epochs before first restart in SGDR",
    )
    train_arg.add_argument(
        "--lr_n_restarts", type=int, default=10, help="Number of restarts in SGDR"
    )
    train_arg.add_argument(
        "--lr_n_mult",
        type=int,
        default=1,
        help="Mult coefficient after restart in SGDR",
    )
    train_arg.add_argument("--lr_end", type=float, default=3 * 1e-4, help="Maximum lr weight value")
    train_arg.add_argument(
        "--n_last", type=int, default=1000, help="Number of iters to smooth loss calc"
    )
    train_arg.add_argument("--n_jobs", type=int, default=1, help="Number of threads")
    train_arg.add_argument(
        "--n_workers", type=int, default=1, help="Number of workers for DataLoaders"
    )

    return parser
