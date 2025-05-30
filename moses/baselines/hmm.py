from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
from pomegranate import DiscreteDistribution, HiddenMarkovModel
from typing_extensions import Self

import moses

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing.pool import Pool

    from numpy.typing import NDArray


class HMM:
    def __init__(
        self,
        n_components: int = 200,
        epochs: int = 100,
        batches_per_epoch: int = 100,
        seed: int = 0,
        verbose: bool = False,
        n_jobs: Pool | int = 1,
    ) -> None:
        """
        Creates a Hidden Markov Model

        Arguments:
            n_components: numebr of states in HMM
            epochs: number of iterations to train model for
            batches_per_epoch: number of batches for minibatch training
            seed: seed for initializing the model
            verbose: if True, will log training statistics
            n_jobs: number of threads for training HMM model
        """
        self.n_components = n_components
        self.epochs = epochs
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.fitted = False

    def fit(self, data: NDArray[np.str_] | Sequence[str]) -> Self:
        """
        Fits a model---learns transition and emission probabilities

        Arguments:
            data: list of SMILES
        """
        list_data = [list(smiles) for smiles in data]
        self.model = HiddenMarkovModel.from_samples(
            DiscreteDistribution,
            n_components=self.n_components,
            end_state=True,
            X=list_data,
            init="kmeans||",
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            max_iterations=self.epochs,
            batches_per_epoch=self.batches_per_epoch,
            random_state=self.seed,
        )
        self.fitted = True
        return self

    def save(self, path: str) -> None:
        """
        Saves a model using pickle

        Arguments:
            path: path to .pkl file for saving
        """
        if not self.fitted:
            raise RuntimeError("Can't save empty model. Fit the model first")
        json = self.model.to_json()
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": json,
                    "n_components": self.n_components,
                    "epochs": self.epochs,
                    "batches_per_epoch": self.batches_per_epoch,
                    "verbose": self.verbose,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> Self:
        """
        Loads saved model

        Arguments:
            path: path to saved .pkl file

        Returns:
            Loaded HMM
        """
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        hmm = data["model"]
        del data["model"]
        model = cls(**data)
        model.model = HiddenMarkovModel.from_json(hmm)
        model.fitted = True
        return model

    def generate_one(self) -> str:
        """
        Generates a SMILES string using a trained HMM

        Retruns:
            SMILES string
        """
        return "".join(self.model.sample())


def reproduce(
    seed: int,
    samples_path: str | None = None,
    metrics_path: str | None = None,
    n_jobs: Pool | int = 1,
    device: str = "cpu",
    verbose: bool = False,
    samples: int = 30000,
) -> tuple[list[str], dict[str, float]]:
    data = moses.get_dataset("train")[:100000]
    if verbose:
        print("Training...")
    model = HMM(n_jobs=n_jobs, seed=seed, verbose=verbose)
    model.fit(data)
    np.random.seed(seed)
    if verbose:
        print(f"Sampling for seed {seed}")
    np.random.seed(seed)
    generated_samples = [model.generate_one() for _ in range(samples)]
    if samples_path is not None:
        with open(samples_path, "w") as f:
            f.write("SMILES\n")
            for sample in generated_samples:
                f.write(sample + "\n")
    if verbose:
        print(f"Computing metrics for seed {seed}")
    metrics = moses.get_all_metrics(generated_samples, n_jobs=n_jobs, device=device)
    if metrics_path is not None:
        with open(metrics_path, "w") as f:
            for key, value in metrics.items():
                f.write("%s,%f\n" % (key, value))
    return generated_samples, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Reproduce HMM experiment for one seed (~24h with n_jobs=32)")
    parser.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        default=1,
        help="Number of threads for computing metrics",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        help="Device for computing metrics",
    )
    parser.add_argument(
        "--samples",
        type=int,
        required=False,
        default=30000,
        help="Number of samples for metrics",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        required=False,
        default=".",
        help="Path to save metrics",
    )
    parser.add_argument("--seed", type=int, required=False, default=1, help="Random seed")
    parser.add_argument("--model_save", type=str, required=False, help="File for saving the model")

    args = parser.parse_known_args()[0]
    reproduce(
        seed=args.seed,
        metrics_path=args.model_save,
        n_jobs=args.n_jobs,
        device=args.device,
        verbose=True,
        samples=args.samples,
    )
