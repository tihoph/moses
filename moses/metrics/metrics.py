from __future__ import annotations

import logging
import multiprocessing
import warnings
from multiprocessing.pool import Pool
from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, TypeVar

import numpy as np
from fcd_torch import FCD as FCDMetric
from scipy.spatial.distance import cosine as cos_distance
from scipy.stats import wasserstein_distance
from typing_extensions import Unpack, override

from moses.dataset import get_dataset, get_statistics
from moses.utils import disable_rdkit_log, enable_rdkit_log, mapper

from .utils import (  # type:ignore[attr-defined]
    QED,
    SA,
    average_agg_tanimoto,
    canonic_smiles,
    compute_fragments,
    compute_scaffolds,
    fingerprints,
    get_mol,
    logP,
    mol_passes_filters,
    weight,
)

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray
    from rdkit import Chem

T = TypeVar("T")
logger = logging.getLogger(__name__)


def get_all_metrics(
    gen: Sequence[str],
    k: int | Sequence[int] | None = None,
    n_jobs: Pool | int = 1,
    device: str = "cpu",
    batch_size: int = 512,
    pool: Pool | int | None = None,
    test: NDArray[np.str_] | Sequence[str] | None = None,
    test_scaffolds: NDArray[np.str_] | Sequence[str] | None = None,
    ptest: Mapping[str, Any] | None = None,
    ptest_scaffolds: Mapping[str, Any] | None = None,
    train: NDArray[np.str_] | Sequence[str] | None = None,
) -> dict[str, float]:
    """
    Computes all available metrics between test (scaffold test)
    and generated sets of SMILES.
    Parameters:
        gen: list of generated SMILES
        k: int or list with values for unique@k. Will calculate number of
            unique molecules in the first k molecules. Default [1000, 10000]
        n_jobs: number of workers for parallel processing
        device: 'cpu' or 'cuda:n', where n is GPU device number
        batch_size: batch size for FCD metric
        pool: optional multiprocessing pool to use for parallelization

        test (None or list): test SMILES. If None, will load
            a default test set
        test_scaffolds (None or list): scaffold test SMILES. If None, will
            load a default scaffold test set
        ptest (None or dict): precalculated statistics of the test set. If
            None, will load default test statistics. If you specified a custom
            test set, default test statistics will be ignored
        ptest_scaffolds (None or dict): precalculated statistics of the
            scaffold test set If None, will load default scaffold test
            statistics. If you specified a custom test set, default test
            statistics will be ignored
        train (None or list): train SMILES. If None, will load a default
            train set
    Available metrics:
        * %valid
        * %unique@k
        * Frechet ChemNet Distance (FCD)
        * Fragment similarity (Frag)
        * Scaffold similarity (Scaf)
        * Similarity to nearest neighbour (SNN)
        * Internal diversity (IntDiv)
        * Internal diversity 2: using square root of mean squared
            Tanimoto similarity (IntDiv2)
        * %passes filters (Filters)
        * Distribution difference for logP, SA, QED, weight
        * Novelty (molecules not present in train)
    """
    if test is None:
        if ptest is not None:
            raise ValueError("You cannot specify custom test statistics for default test set")
        test = get_dataset("test")
        ptest = get_statistics("test")

    if test_scaffolds is None:
        if ptest_scaffolds is not None:
            raise ValueError(
                "You cannot specify custom scaffold test statistics for default scaffold test set"
            )
        test_scaffolds = get_dataset("test_scaffolds")
        ptest_scaffolds = get_statistics("test_scaffolds")

    train = train or get_dataset("train")

    if k is None:
        k = [1000, 10000]
    disable_rdkit_log()
    metrics: dict[str, float] = {}
    close_pool = False
    if isinstance(n_jobs, Pool):
        pool = n_jobs
        n_jobs = len(pool._pool)  # type: ignore[attr-defined] # noqa: SLF001
    if not isinstance(pool, Pool):
        if n_jobs != 1:
            pool = multiprocessing.Pool(pool or n_jobs)
            close_pool = True
        else:
            pool = 1
    metrics["valid"] = fraction_valid(gen, n_jobs=pool)
    gen = remove_invalid(gen, canonize=True)
    if isinstance(k, int):
        k = [k]
    for _k in k:
        metrics["unique@{}".format(_k)] = fraction_unique(gen, _k, pool)

    if ptest is None:
        ptest = compute_intermediate_statistics(
            test, n_jobs=n_jobs, device=device, batch_size=batch_size, pool=pool
        )
    if test_scaffolds is not None and ptest_scaffolds is None:  # type: ignore[redundant-expr]
        ptest_scaffolds = compute_intermediate_statistics(
            test_scaffolds,
            n_jobs=n_jobs,
            device=device,
            batch_size=batch_size,
            pool=pool,
        )
    raw_mols = mapper(pool)(get_mol, gen)
    mols = [m for m in raw_mols if m is not None]
    if len(mols) != len(raw_mols):
        raise ValueError("Some molecules could not be mapped to RDKit molecules")

    kwargs: MetricsKwargs = {"n_jobs": pool, "device": device, "batch_size": batch_size}
    kwargs_fcd: FCDKwargs = {
        "n_jobs": n_jobs,
        "device": device,
        "batch_size": batch_size,
    }
    metrics["FCD/Test"] = FCDMetric(**kwargs_fcd)(gen=gen, pref=ptest["FCD"])
    metrics["SNN/Test"] = SNNMetric(**kwargs)(gen=mols, pref=ptest["SNN"])
    metrics["Frag/Test"] = FragMetric(**kwargs)(gen=mols, pref=ptest["Frag"])
    metrics["Scaf/Test"] = ScafMetric(**kwargs)(gen=mols, pref=ptest["Scaf"])
    if ptest_scaffolds is not None:
        metrics["FCD/TestSF"] = FCDMetric(**kwargs_fcd)(gen=gen, pref=ptest_scaffolds["FCD"])
        metrics["SNN/TestSF"] = SNNMetric(**kwargs)(gen=mols, pref=ptest_scaffolds["SNN"])
        metrics["Frag/TestSF"] = FragMetric(**kwargs)(gen=mols, pref=ptest_scaffolds["Frag"])
        metrics["Scaf/TestSF"] = ScafMetric(**kwargs)(gen=mols, pref=ptest_scaffolds["Scaf"])

    metrics["IntDiv"] = internal_diversity(mols, pool, device=device)
    metrics["IntDiv2"] = internal_diversity(mols, pool, device=device, p=2)
    metrics["Filters"] = fraction_passes_filters(mols, pool)

    # Properties
    for name, func in [("logP", logP), ("SA", SA), ("QED", QED), ("weight", weight)]:
        metrics[name] = WassersteinMetric(func, **kwargs)(gen=mols, pref=ptest[name])

    if train is not None:
        metrics["Novelty"] = novelty(mols, train, pool)
    enable_rdkit_log()
    if close_pool and isinstance(pool, Pool):
        pool.close()
        pool.join()
    return metrics


def compute_intermediate_statistics(
    smiles: NDArray[np.str_] | Sequence[str],
    n_jobs: Pool | int = 1,
    device: str = "cpu",
    batch_size: int = 512,
    pool: Pool | int | None = None,
) -> dict[str, Any]:
    """
    The function precomputes statistics such as mean and variance for FCD, etc.
    It is useful to compute the statistics for test and scaffold test sets to
        speedup metrics calculation.
    """
    close_pool = False
    if isinstance(n_jobs, Pool):
        pool = n_jobs
        n_jobs = len(pool._pool)  # type: ignore[attr-defined] # noqa: SLF001
    if not isinstance(pool, Pool):
        if n_jobs != 1:
            pool = multiprocessing.Pool(pool or n_jobs)
            close_pool = True
        else:
            pool = 1
    statistics: dict[str, Any] = {}
    raw_mols = mapper(pool)(get_mol, smiles)
    mols = [m for m in raw_mols if m is not None]
    if len(mols) != len(raw_mols):
        raise ValueError("Some molecules could not be mapped to RDKit molecules")

    kwargs: MetricsKwargs = {"n_jobs": pool, "device": device, "batch_size": batch_size}
    kwargs_fcd: FCDKwargs = {
        "n_jobs": n_jobs,
        "device": device,
        "batch_size": batch_size,
    }
    statistics["FCD"] = FCDMetric(**kwargs_fcd).precalc(smiles)
    statistics["SNN"] = SNNMetric(**kwargs).precalc(mols)
    statistics["Frag"] = FragMetric(**kwargs).precalc(mols)
    statistics["Scaf"] = ScafMetric(**kwargs).precalc(mols)
    for name, func in [("logP", logP), ("SA", SA), ("QED", QED), ("weight", weight)]:
        statistics[name] = WassersteinMetric(func, **kwargs).precalc(mols)
    if close_pool and isinstance(pool, Pool):
        pool.terminate()
    return statistics


def fraction_passes_filters(gen: Sequence[Chem.Mol], n_jobs: Pool | int = 1) -> float:
    """
    Computes the fraction of molecules that pass filters:
    * MCF
    * PAINS
    * Only allowed atoms ('C','N','S','O','F','Cl','Br','H')
    * No charges
    """
    passes = mapper(n_jobs)(mol_passes_filters, gen)
    return np.mean(passes)  # type: ignore[return-value]


def internal_diversity(
    gen: Sequence[str | Chem.Mol],
    n_jobs: Pool | int = 1,
    device: str = "cpu",
    fp_type: str = "morgan",
    gen_fps: NDArray[Any] | None = None,
    p: int = 1,
) -> float:
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    avg_sim = average_agg_tanimoto(gen_fps, gen_fps, agg="mean", device=device, p=p)
    return 1 - avg_sim


def fraction_unique(
    gen: Sequence[str],
    k: int | None = None,
    n_jobs: Pool | int = 1,
    check_validity: bool = True,
) -> float:
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            warnings.warn(  # noqa: B028
                "Can't compute unique@{}.".format(k)
                + "gen contains only {} molecules".format(len(gen))
            )
        gen = gen[:k]
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return len(canonic) / len(gen)


def fraction_valid(gen: Sequence[str], n_jobs: Pool | int = 1) -> float:
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    mols = mapper(n_jobs)(get_mol, gen)
    return 1 - mols.count(None) / len(mols)


def novelty(
    gen: Sequence[str | Chem.Mol],
    train: NDArray[np.str_] | Sequence[str],
    n_jobs: Pool | int = 1,
) -> float:
    gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return len(gen_smiles_set - train_set) / len(gen_smiles_set)


def remove_invalid(
    gen: NDArray[np.str_] | Sequence[str], canonize: bool = True, n_jobs: Pool | int = 1
) -> list[str]:
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if x is not None]


class MetricsKwargs(TypedDict, total=False):
    n_jobs: Pool | int
    device: str
    batch_size: int


class FCDKwargs(TypedDict, total=False):
    n_jobs: int
    device: str
    batch_size: int


class Metric:
    def __init__(
        self,
        n_jobs: Pool | int = 1,
        device: str = "cpu",
        batch_size: int = 512,
        # **kwargs,
    ) -> None:
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        # for k, v in kwargs.values():
        #     setattr(self, k, v) # noqa: ERA001

    def __call__(
        self,
        ref: Sequence[Chem.Mol] | None = None,
        gen: Sequence[Chem.Mol] | None = None,
        pref: Mapping[str, Any] | None = None,
        pgen: Mapping[str, Any] | None = None,
    ) -> float:
        if pref is None:
            if ref is None:
                raise ValueError("specify ref or pref")
            pref = self.precalc(ref)
        elif ref:
            raise ValueError("specify pref xor ref")

        if pgen is None:
            if gen is None:
                raise ValueError("specify gen or pgen")
            pgen = self.precalc(gen)
        elif gen:
            raise ValueError("specify pgen xor gen")

        return self.metric(pref, pgen)

    def precalc(self, mols: Sequence[Chem.Mol]) -> dict[str, Any]:
        raise NotImplementedError

    def metric(self, pref: Mapping[str, Any], pgen: Mapping[str, Any]) -> float:
        raise NotImplementedError


class SNNMetric(Metric):
    """
    Computes average max similarities of gen SMILES to ref SMILES
    """

    def __init__(
        self,
        fp_type: Literal["maccs", "morgan"] = "morgan",
        **kwargs: Unpack[MetricsKwargs],
    ) -> None:
        self.fp_type = fp_type
        super().__init__(**kwargs)

    @override
    def precalc(self, mols: Sequence[Chem.Mol]) -> dict[str, NDArray[Any]]:
        return {"fps": fingerprints(mols, n_jobs=self.n_jobs, fp_type=self.fp_type)}

    @override
    def metric(self, pref: Mapping[str, Any], pgen: Mapping[str, Any]) -> float:
        return average_agg_tanimoto(pref["fps"], pgen["fps"], device=self.device)


def cos_similarity(ref_counts: Mapping[str, int], gen_counts: Mapping[str, int]) -> float:
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:

     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)  # type: ignore[no-any-return]


class FragMetric(Metric):
    @override
    def precalc(self, mols: Sequence[Chem.Mol]) -> dict[str, Counter[str]]:
        return {"frag": compute_fragments(mols, n_jobs=self.n_jobs)}

    @override
    def metric(self, pref: Mapping[str, Any], pgen: Mapping[str, Any]) -> float:
        return cos_similarity(pref["frag"], pgen["frag"])


class ScafMetric(Metric):
    @override
    def precalc(self, mols: Sequence[Chem.Mol]) -> dict[str, Counter[str]]:
        return {"scaf": compute_scaffolds(mols, n_jobs=self.n_jobs)}

    @override
    def metric(self, pref: Mapping[str, Any], pgen: Mapping[str, Any]) -> float:
        return cos_similarity(pref["scaf"], pgen["scaf"])


class WassersteinMetric(Metric, Generic[T]):
    def __init__(
        self,
        func: Callable[[Chem.Mol], T] | None = None,
        **kwargs: Unpack[MetricsKwargs],
    ):
        self.func = func
        super().__init__(**kwargs)

    @override
    def precalc(self, mols: Sequence[Chem.Mol]) -> dict[str, list[T] | list[Chem.Mol]]:
        if self.func is not None:
            values = mapper(self.n_jobs)(self.func, mols)
        else:
            values = list(mols)
        return {"values": values}

    @override
    def metric(self, pref: Mapping[str, Any], pgen: Mapping[str, Any]) -> float:
        return wasserstein_distance(pref["values"], pgen["values"])  # type: ignore[no-any-return]
