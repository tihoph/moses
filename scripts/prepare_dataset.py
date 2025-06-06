from __future__ import annotations

import argparse
import gzip
import logging
from functools import partial
from multiprocessing import Pool
from typing import TYPE_CHECKING

import pandas as pd
from rdkit import Chem
from tqdm.auto import tqdm

from moses.metrics import compute_scaffold, mol_passes_filters

if TYPE_CHECKING:
    from collections.abc import Sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prepare dataset")


class PrepareConfig(argparse.Namespace):
    output: str = "dataset_v1.csv"
    seed: int = 0
    zinc: str = "../data/11_p0.smi.gz"
    n_jobs: int = 1
    keep_ids: int = False
    isomeric: bool = False


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=PrepareConfig.output,
        help="Path for constructed dataset",
    )
    parser.add_argument("--seed", type=int, default=PrepareConfig.seed, help="Random state")
    parser.add_argument(
        "--zinc",
        type=str,
        default=PrepareConfig.zinc,
        help="path to .smi.gz file with ZINC smiles",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=PrepareConfig.n_jobs, help="number of processes to use"
    )
    parser.add_argument(
        "--keep_ids",
        action="store_true",
        default=PrepareConfig.keep_ids,
        help="Keep ZINC ids in the final csv file",
    )
    parser.add_argument(
        "--isomeric",
        action="store_true",
        default=PrepareConfig.isomeric,
        help="Save isomeric SMILES (non-isomeric by default)",
    )
    return parser


def process_molecule(mol_row: bytes, isomeric: bool) -> tuple[str, str] | None:
    mol_row_str = mol_row.decode("utf-8")
    smiles, _id = mol_row_str.split()
    if not mol_passes_filters(smiles):
        return None
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=isomeric)
    return _id, smiles


def unzip_dataset(path: str) -> list[bytes]:
    logger.info("Unzipping dataset")
    with gzip.open(path) as smi:
        return smi.readlines()


def filter_lines(lines: Sequence[bytes], n_jobs: int, isomeric: bool) -> pd.DataFrame:
    logger.info("Filtering SMILES")
    with Pool(n_jobs) as pool:
        process_molecule_p = partial(process_molecule, isomeric=isomeric)
        dataset_ls = [
            x
            for x in tqdm(
                pool.imap_unordered(process_molecule_p, lines),
                total=len(lines),
                miniters=1000,
            )
            if x is not None
        ]
        dataset = pd.DataFrame(dataset_ls, columns=["ID", "SMILES"])
        dataset = dataset.sort_values(by=["ID", "SMILES"])
        dataset = dataset.drop_duplicates("ID")
        dataset = dataset.sort_values(by="ID")
        dataset = dataset.drop_duplicates("SMILES")
        dataset["scaffold"] = pool.map(compute_scaffold, dataset["SMILES"].values)
    return dataset


def split_dataset(dataset: pd.DataFrame, seed: int) -> pd.DataFrame:
    logger.info("Splitting the dataset")
    scaffolds = pd.value_counts(dataset["scaffold"])
    scaffolds = sorted(scaffolds.items(), key=lambda x: (-x[1], x[0]))
    test_scaffolds = {x[0] for x in scaffolds[9::10]}
    dataset["SPLIT"] = "train"
    test_scaf_idx = [x in test_scaffolds for x in dataset["scaffold"]]
    dataset.loc[test_scaf_idx, "SPLIT"] = "test_scaffolds"
    test_idx = dataset.loc[dataset["SPLIT"] == "train"].sample(frac=0.1, random_state=seed).index
    dataset.loc[test_idx, "SPLIT"] = "test"
    return dataset.drop("scaffold", axis=1)


def main(config: PrepareConfig) -> None:
    lines = unzip_dataset(config.zinc)
    dataset = filter_lines(lines, config.n_jobs, config.isomeric)
    dataset = split_dataset(dataset, config.seed)
    if not config.keep_ids:
        dataset = dataset.drop("ID", 1)
    dataset.to_csv(config.output, index=None)


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_args(namespace=PrepareConfig())
    main(config)
