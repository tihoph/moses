from __future__ import annotations

import os
from collections import Counter
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import pandas as pd
import scipy.sparse
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.AllChem import (  # type: ignore[attr-defined]
    GetMorganFingerprintAsBitVect as Morgan,
)
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds import MurckoScaffold

from moses.metrics.NP_Score import npscorer
from moses.metrics.SA_Score import sascorer
from moses.utils import get_mol, mapper

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from multiprocessing.pool import Pool

    from numpy.typing import NDArray

DTypeT = TypeVar("DTypeT", bound=np.generic)


_base_dir = os.path.split(__file__)[0]
_mcf = pd.read_csv(os.path.join(_base_dir, "mcf.csv"))
_pains = pd.read_csv(os.path.join(_base_dir, "wehi_pains.csv"), names=["smarts", "names"])
_combined = pd.concat([_mcf, _pains], sort=True)
_filters = [Chem.MolFromSmarts(x) for x in _combined["smarts"].to_numpy()]


def canonic_smiles(smiles_or_mol: str | Chem.Mol) -> str | None:
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def logP(mol: Chem.Mol) -> float:
    """
    Computes RDKit's logP
    """
    return Chem.Crippen.MolLogP(mol)  # type: ignore[attr-defined,no-any-return]


def SA(mol: Chem.Mol) -> float:
    """
    Computes RDKit's Synthetic Accessibility score
    """
    return sascorer.calculateScore(mol)


def NP(mol: Chem.Mol) -> float:
    """
    Computes RDKit's Natural Product-likeness score
    """
    return npscorer.scoreMol(mol)


def QED(mol: Chem.Mol) -> float:
    """
    Computes RDKit's QED score
    """
    return qed(mol)  # type: ignore[no-untyped-call,no-any-return]


def weight(mol: Chem.Mol) -> float:
    """
    Computes molecular weight for given molecule.
    Returns float,
    """
    return Descriptors.MolWt(mol)  # type: ignore[attr-defined,no-any-return]


def get_n_rings(mol: Chem.Mol) -> int:
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()


def fragmenter(mol: str | Chem.Mol) -> list[str]:
    """
    fragment mol using BRICS and return smiles list
    """
    fgs = AllChem.FragmentOnBRICSBonds(get_mol(mol))  # type: ignore[attr-defined]
    return Chem.MolToSmiles(fgs).split(".")


def compute_fragments(mol_list: Sequence[Chem.Mol], n_jobs: Pool | int = 1) -> Counter[str]:
    """
    fragment list of mols using BRICS and return smiles list
    """
    fragments: Counter[str] = Counter()
    for mol_frag in mapper(n_jobs)(fragmenter, mol_list):
        fragments.update(mol_frag)
    return fragments


def compute_scaffolds(
    mol_list: Sequence[Chem.Mol], n_jobs: Pool | int = 1, min_rings: int = 2
) -> Counter[str]:
    """
    Extracts a scafold from a molecule in a form of a canonic SMILES
    """
    map_ = mapper(n_jobs)
    scaffolds = map_(partial(compute_scaffold, min_rings=min_rings), mol_list)
    return Counter(scaf for scaf in scaffolds if scaf is not None)


def compute_scaffold(mol: Chem.Mol, min_rings: int = 2) -> str | None:
    proc_mol = get_mol(mol)
    if not proc_mol:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(proc_mol)  # type: ignore[no-untyped-call]
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == "" or n_rings < min_rings:
        return None
    return scaffold_smiles


def average_agg_tanimoto(
    stock_vecs: NDArray[np.floating[Any]],
    gen_vecs: NDArray[np.floating[Any]],
    batch_size: int = 5000,
    agg: Literal["max", "mean"] = "max",
    device: str = "cpu",
    p: float = 1,
) -> float:
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ["max", "mean"], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j : j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i : i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (
                (tp / (x_stock.sum(1, keepdim=True) + y_gen.sum(0, keepdim=True) - tp))
                .cpu()
                .numpy()
            )
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == "max":
                agg_tanimoto[i : i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i : i + y_gen.shape[1]], jac.max(0)
                )
            elif agg == "mean":
                agg_tanimoto[i : i + y_gen.shape[1]] += jac.sum(0)
                total[i : i + y_gen.shape[1]] += jac.shape[0]
    if agg == "mean":
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto) ** (1 / p)
    return np.mean(agg_tanimoto)  # type:ignore[return-value]


def fingerprint(
    smiles_or_mol: str | Chem.Mol,
    fp_type: Literal["maccs", "morgan"] = "maccs",
    dtype: type[DTypeT] | None = None,
    morgan__r: int = 2,
    morgan__n: int = 1024,
) -> NDArray[np.uint8] | NDArray[DTypeT] | None:
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits

    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    if fp_type == "maccs":
        keys = MACCSkeys.GenMACCSKeys(mol)  # type:ignore[attr-defined]
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype="uint8")
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == "morgan":
        fingerprint = np.asarray(Morgan(mol, morgan__r, nBits=morgan__n), dtype="uint8")
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint


def fingerprints(
    smiles_mols_array: NDArray[np.str_] | pd.Series[str] | Sequence[Chem.Mol | str],
    n_jobs: Pool | int = 1,
    already_unique: bool = False,
    *args: Any,
    **kwargs: Any,
) -> NDArray[Any]:
    """
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    """
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.to_numpy()
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    inv_index: NDArray[np.intp] | None = None
    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array, return_inverse=True)

    fps = mapper(n_jobs)(partial(fingerprint, *args, **kwargs), smiles_mols_array)

    first_fp: NDArray[np.uint8] | NDArray[Any] | None = None
    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps_ls = [fp if fp is not None else np.array([np.nan]).repeat(length)[None, :] for fp in fps]

    if first_fp is None:
        raise ValueError("No valid fingerprint found")

    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps_ls).tocsr()
    else:
        fps = np.vstack(fps_ls)  # type:ignore[assignment]
    if not already_unique:
        if inv_index is None:
            raise RuntimeError("inv_index not calculated")
        return fps[inv_index]  # type:ignore[return-value]
    return fps  # type:ignore[return-value]


def mol_passes_filters(
    mol: str | Chem.Mol,
    allowed: Collection[str] | None = None,
    isomericSmiles: bool = False,
) -> bool:
    """
    Checks if mol
    * passes MCF and PAINS filters,
    * has only allowed atoms
    * is not charged
    """
    allowed = allowed or {"C", "N", "S", "O", "F", "Cl", "Br", "H"}
    proc_mol = get_mol(mol)
    if proc_mol is None:
        return False
    ring_info = proc_mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(len(x) >= 8 for x in ring_info.AtomRings()):
        return False
    h_mol = Chem.AddHs(mol)
    if any(atom.GetFormalCharge() != 0 for atom in proc_mol.GetAtoms()):  # type:ignore[call-arg,no-untyped-call]
        return False
    if any(atom.GetSymbol() not in allowed for atom in proc_mol.GetAtoms()):  # type:ignore[call-arg,no-untyped-call]
        return False
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
        return False
    smiles = Chem.MolToSmiles(proc_mol, isomericSmiles=isomericSmiles)
    if not smiles:
        return False
    return Chem.MolFromSmiles(smiles) is not None
