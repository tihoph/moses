from __future__ import annotations

import argparse
import random
import re
from argparse import _ActionsContainer

import numpy as np
import pandas as pd
import torch


class CommonConfig(argparse.Namespace):
    device: str = "cuda"
    seed: int = 0


class TrainConfig(CommonConfig):
    train_load: str | None
    val_load: str | None
    model_save: str = "model.pt"
    save_frequency: int = 20
    log_file: str
    config_save: str
    vocab_save: str | None
    vocab_load: str | None


class SampleConfig(CommonConfig):
    model_load: str
    config_load: str
    vocab_load: str
    n_samples: int
    gen_save: str
    n_batch: int = 32
    max_len: int = 100


def add_common_arg(parser: _ActionsContainer) -> _ActionsContainer:
    def torch_device(arg: str) -> str:
        if re.match("^(cuda(:[0-9]+)?|cpu)$", arg) is None:
            raise argparse.ArgumentTypeError("Wrong device format: {}".format(arg))

        if arg != "cpu":
            splited_device = arg.split(":")

            if (not torch.cuda.is_available()) or (
                len(splited_device) > 1 and int(splited_device[1]) > torch.cuda.device_count()
            ):
                raise argparse.ArgumentTypeError("Wrong device: {} is not available".format(arg))

        return arg

    # Base
    parser.add_argument(
        "--device",
        type=torch_device,
        default=CommonConfig.device,
        help='Device to run: "cpu" or "cuda:<device number>"',
    )
    parser.add_argument("--seed", type=int, default=CommonConfig.seed, help="Seed")

    return parser


def add_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Common
    common_arg = parser.add_argument_group("Common")
    add_common_arg(common_arg)
    common_arg.add_argument("--train_load", type=str, help="Input data in csv format to train")
    common_arg.add_argument("--val_load", type=str, help="Input data in csv format to validation")
    common_arg.add_argument(
        "--model_save",
        type=str,
        required=True,
        default=TrainConfig.model_save,
        help="Where to save the model",
    )
    common_arg.add_argument(
        "--save_frequency",
        type=int,
        default=TrainConfig.save_frequency,
        help="How often to save the model",
    )
    common_arg.add_argument("--log_file", type=str, required=False, help="Where to save the log")
    common_arg.add_argument(
        "--config_save", type=str, required=True, help="Where to save the config"
    )
    common_arg.add_argument("--vocab_save", type=str, help="Where to save the vocab")
    common_arg.add_argument(
        "--vocab_load",
        type=str,
        help="Where to load the vocab; otherwise it will be evaluated",
    )

    return parser


def add_sample_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Common
    common_arg = parser.add_argument_group("Common")
    add_common_arg(common_arg)
    common_arg.add_argument("--model_load", type=str, required=True, help="Where to load the model")
    common_arg.add_argument(
        "--config_load", type=str, required=True, help="Where to load the config"
    )
    common_arg.add_argument("--vocab_load", type=str, required=True, help="Where to load the vocab")
    common_arg.add_argument(
        "--n_samples", type=int, required=True, help="Number of samples to sample"
    )
    common_arg.add_argument(
        "--gen_save", type=str, required=True, help="Where to save the gen molecules"
    )
    common_arg.add_argument(
        "--n_batch", type=int, default=SampleConfig.n_batch, help="Size of batch"
    )
    common_arg.add_argument(
        "--max_len", type=int, default=SampleConfig.max_len, help="Max of length of SMILES"
    )

    return parser


def read_smiles_csv(path: str) -> list[str]:
    return pd.read_csv(path, usecols=["SMILES"], squeeze=True).astype(str).tolist()  # type: ignore[no-any-return]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
