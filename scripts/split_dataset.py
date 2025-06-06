import argparse
import os

import numpy as np
import pandas as pd

from moses.metrics import compute_intermediate_statistics


class SplitConfig(argparse.Namespace):
    dir: str = "./data"
    no_subset: bool
    train_size: int
    test_size: int
    seed: int = 0
    precompute: bool = True
    n_jobs: int = 1
    device: str = "cpu"
    batch_size: int = 512


def str2bool(v: str) -> bool:
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir", type=str, default=SplitConfig.dir, help="Directory for splitted dataset"
    )
    parser.add_argument(
        "--no_subset",
        action="store_true",
        help="Do not create subsets for training and testing",
    )
    parser.add_argument("--train_size", type=int, help="Size of training dataset")
    parser.add_argument("--test_size", type=int, help="Size of testing dataset")
    parser.add_argument("--seed", type=int, default=SplitConfig.seed, help="Random state")
    parser.add_argument(
        "--precompute",
        type=str2bool,
        default=SplitConfig.precompute,
        help="Precompute intermediate statistics",
    )
    parser.add_argument("--n_jobs", type=int, default=SplitConfig.n_jobs, help="Number of workers")
    parser.add_argument("--device", type=str, default=SplitConfig.device, help="GPU device id")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=SplitConfig.batch_size,
        help="Batch size for FCD calculation",
    )
    return parser


def main(config: SplitConfig) -> None:
    dataset_path = os.path.join(config.dir, "dataset_v1.csv")
    repo_url = "https://media.githubusercontent.com/media/molecularsets/moses/"
    download_url = repo_url + "master/data/dataset_v1.csv"
    if not os.path.exists(dataset_path):
        raise ValueError(
            "Missing dataset_v1.csv in {}; ".format(config.dir)
            + "Please, use 'git lfs pull' or download it manually from "
            + download_url
        )

    if config.no_subset:
        return

    data = pd.read_csv(dataset_path)

    train_data = data[data["SPLIT"] == "train"]
    test_data = data[data["SPLIT"] == "test"]
    test_scaffolds_data = data[data["SPLIT"] == "test_scaffolds"]

    if config.train_size is not None:
        train_data = train_data.sample(config.train_size, random_state=config.seed)

    if config.test_size is not None:
        test_data = test_data.sample(config.test_size, random_state=config.seed)
        test_scaffolds_data = test_scaffolds_data.sample(config.test_size, random_state=config.seed)

    train_data.to_csv(os.path.join(config.dir, "train.csv"), index=False)
    test_data.to_csv(os.path.join(config.dir, "test.csv"), index=False)
    test_scaffolds_data.to_csv(os.path.join(config.dir, "test_scaffolds.csv"), index=False)

    if config.precompute:
        test_stats = compute_intermediate_statistics(
            test_data["SMILES"].values,
            n_jobs=config.n_jobs,
            device=config.device,
            batch_size=config.batch_size,
        )
        test_sf_stats = compute_intermediate_statistics(
            test_scaffolds_data["SMILES"].values,
            n_jobs=config.n_jobs,
            device=config.device,
            batch_size=config.batch_size,
        )
        np.savez(os.path.join(config.dir, "test_stats.npz"), stats=test_stats)  # type: ignore[arg-type]
        np.savez(
            os.path.join(config.dir, "test_scaffolds_stats.npz"),
            stats=test_sf_stats,  # type: ignore[arg-type]
        )


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_args(namespace=SplitConfig())
    main(config)
