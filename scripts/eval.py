from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from rdkit import RDLogger

from moses.metrics.metrics import get_all_metrics
from moses.script_utils import read_smiles_csv

if TYPE_CHECKING:
    from collections.abc import Sequence

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # type: ignore[no-untyped-call]


class EvalConfig(argparse.Namespace):
    test_path: str
    test_scaffolds_path: str
    train_path: str
    ptest_path: str
    ptest_scaffolds_path: str
    gen_path: str
    unique_k: Sequence[int] = (1000, 10000)
    n_jobs: int = 1
    device: str = "cpu"


@overload
def main(config: EvalConfig, print_metrics: Literal[True] = True) -> None: ...


@overload
def main(config: EvalConfig, print_metrics: Literal[False] = ...) -> dict[str, float]: ...


def main(config: EvalConfig, print_metrics: bool = True) -> dict[str, float] | None:
    test = None
    test_scaffolds = None
    ptest = None
    ptest_scaffolds = None
    train = None
    if config.test_path:
        test = read_smiles_csv(config.test_path)
    if config.test_scaffolds_path is not None:
        test_scaffolds = read_smiles_csv(config.test_scaffolds_path)
    if config.train_path is not None:
        train = read_smiles_csv(config.train_path)
    if config.ptest_path is not None:
        ptest = np.load(config.ptest_path, allow_pickle=True)["stats"].item()
    if config.ptest_scaffolds_path is not None:
        ptest_scaffolds = np.load(config.ptest_scaffolds_path, allow_pickle=True)["stats"].item()
    gen = read_smiles_csv(config.gen_path)
    metrics = get_all_metrics(
        gen=gen,
        k=config.ks,
        n_jobs=config.n_jobs,
        device=config.device,
        test_scaffolds=test_scaffolds,
        ptest=ptest,
        ptest_scaffolds=ptest_scaffolds,
        test=test,
        train=train,
    )

    if print_metrics:
        for name, value in metrics.items():
            print("{},{}".format(name, value))
        return None
    return metrics


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, required=False, help="Path to test molecules csv")
    parser.add_argument(
        "--test_scaffolds_path",
        type=str,
        required=False,
        help="Path to scaffold test molecules csv",
    )
    parser.add_argument(
        "--train_path", type=str, required=False, help="Path to train molecules csv"
    )
    parser.add_argument(
        "--ptest_path", type=str, required=False, help="Path to precalculated test npz"
    )
    parser.add_argument(
        "--ptest_scaffolds_path",
        type=str,
        required=False,
        help="Path to precalculated scaffold test npz",
    )
    parser.add_argument(
        "--gen_path", type=str, required=True, help="Path to generated molecules csv"
    )
    parser.add_argument(
        "--ks",
        "--unique_k",
        nargs="+",
        default=EvalConfig.unique_k,
        type=int,
        help="Number of molecules to calculate uniqueness at."
        "Multiple values are possible. Defaults to "
        "--unique_k 1000 10000",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=EvalConfig.n_jobs, help="Number of processes to run metrics"
    )
    parser.add_argument(
        "--device", type=str, default=EvalConfig.device, help="GPU device id (`cpu` or `cuda:n`)"
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_known_args(namespace=EvalConfig())[0]
    main(config)
