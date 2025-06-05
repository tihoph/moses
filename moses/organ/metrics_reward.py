from __future__ import annotations

import random
from collections import Counter
from multiprocessing.pool import Pool
from typing import TYPE_CHECKING

import numpy as np

from moses.metrics import (
    QED,
    SA,
    FCDMetric,
    FragMetric,
    ScafMetric,
    SNNMetric,
    WassersteinMetric,
    fraction_passes_filters,
    internal_diversity,
    logP,
    remove_invalid,
    weight,
)
from moses.utils import get_mol, mapper

from .config import SUPPORTED_METRICS, MetricT

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from numpy.typing import NDArray
    from rdkit import Chem


class MetricsReward:
    @staticmethod
    def _nan2zero(value: float) -> float:
        if value == np.nan:
            return 0

        return value

    def __init__(
        self,
        n_ref_subsample: int,
        n_rollouts: int,
        n_jobs: Pool | int,
        metrics: Collection[MetricT] | None = None,
    ) -> None:
        self.n_ref_subsample = n_ref_subsample
        self.n_rollouts = n_rollouts
        self.n_jobs = Pool(n_jobs) if isinstance(n_jobs, int) and n_jobs > 1 else n_jobs
        self.metrics = []
        if metrics:
            assert all(m in SUPPORTED_METRICS for m in metrics)
            self.metrics = list(metrics)

    def get_reference_data(
        self, data: NDArray[np.str_] | Sequence[str]
    ) -> tuple[list[str], list[Chem.Mol]]:
        ref_smiles = remove_invalid(data, canonize=True, n_jobs=self.n_jobs)
        ref_mols = mapper(self.n_jobs)(get_mol, ref_smiles)
        valid_ref_mols = [m for m in ref_mols if m is not None]
        if len(valid_ref_mols) != len(ref_mols):
            raise ValueError("Invalid molecules in reference data")
        return ref_smiles, valid_ref_mols

    def _get_metrics(
        self, ref: Sequence[str], ref_mols: Sequence[Chem.Mol], rollout: Sequence[str]
    ) -> list[list[float]]:
        raw_rollout_mols = mapper(self.n_jobs)(get_mol, rollout)
        result = [[0.0 if m is None else 1.0] for m in raw_rollout_mols]

        if sum([r[0] for r in result], 0) == 0:
            return result

        rollout = remove_invalid(rollout, canonize=True, n_jobs=self.n_jobs)
        raw_rollout_mols = mapper(self.n_jobs)(get_mol, rollout)
        rollout_mols = [m for m in raw_rollout_mols if m is not None]
        if len(rollout_mols) != len(raw_rollout_mols):
            raise ValueError("Some molecules could not be mapped to RDKit molecules")

        if len(rollout) < 2:
            return result

        if len(self.metrics):
            for metric_name in self.metrics:
                if metric_name == "fcd":
                    m = FCDMetric(n_jobs=self.n_jobs)(ref, rollout)
                elif metric_name == "morgan":  # type:ignore[comparison-overlap]
                    m = SNNMetric(n_jobs=self.n_jobs)(ref_mols, rollout_mols)  # type: ignore[unreachable]
                elif metric_name == "fragments":
                    m = FragMetric(n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == "scaffolds":
                    m = ScafMetric(n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == "internal_diversity":
                    m = internal_diversity(rollout_mols, n_jobs=self.n_jobs)
                elif metric_name == "filters":
                    m = fraction_passes_filters(rollout_mols, n_jobs=self.n_jobs)
                elif metric_name == "logp":
                    m = -WassersteinMetric(func=logP, n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == "sa":
                    m = -WassersteinMetric(func=SA, n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == "qed":
                    m = -WassersteinMetric(func=QED, n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == "weight":
                    m = -WassersteinMetric(func=weight, n_jobs=self.n_jobs)(ref_mols, rollout_mols)

                m = MetricsReward._nan2zero(m)
                for i in range(len(rollout)):
                    result[i].append(m)

        return result

    def __call__(
        self, gen: Sequence[str], ref: Sequence[str], ref_mols: Sequence[Chem.Mol]
    ) -> list[float]:
        idxs = random.sample(range(len(ref)), self.n_ref_subsample)
        ref_subsample = [ref[idx] for idx in idxs]
        ref_mols_subsample = [ref_mols[idx] for idx in idxs]

        gen_counter = Counter(gen)
        gen_counts = [gen_counter[g] for g in gen]

        n = len(gen) // self.n_rollouts
        rollouts = [gen[i::n] for i in range(n)]

        metrics_values = [
            self._get_metrics(ref_subsample, ref_mols_subsample, rollout) for rollout in rollouts
        ]
        metrics_values_map = map(  # noqa: C417
            lambda rm: [sum(r, 0) / len(r) for r in rm], metrics_values
        )
        reward_values: tuple[float, ...] = sum(zip(*metrics_values_map), ())
        return [v / c for v, c in zip(reward_values, gen_counts)]
