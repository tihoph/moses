import os
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SplitT = Literal["train", "test", "test_scaffolds"]
AVAILABLE_SPLITS: list[SplitT] = list(get_args(SplitT))


def get_dataset(split: SplitT = "train") -> NDArray[np.str_]:
    """
    Loads MOSES dataset

    Arguments:
        split (str): split to load. Must be
            one of: 'train', 'test', 'test_scaffolds'

    Returns:
        list with SMILES strings
    """
    if split not in AVAILABLE_SPLITS:
        raise ValueError(f"Unknown split {split}. Available splits: {AVAILABLE_SPLITS}")
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, "data", split + ".csv.gz")
    return pd.read_csv(path, compression="gzip")["SMILES"].to_numpy()  # type: ignore[no-any-return]


def get_statistics(split: SplitT = "test") -> dict[str, Any]:
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, "data", split + "_stats.npz")
    return np.load(path, allow_pickle=True)["stats"].item()  # type: ignore[no-any-return]
