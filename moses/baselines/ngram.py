from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm
from typing_extensions import Self

import moses
from moses import CharVocab

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing.pool import Pool

    from numpy.typing import NDArray


class NGram:
    def __init__(self, max_context_len: int = 10, verbose: bool = False) -> None:
        self.max_context_len = max_context_len
        self._dict: dict[tuple[int, ...], NDArray[np.float64]] = {}
        self.vocab: CharVocab | None = None
        self.default_probs: NDArray[np.float64] | None = None
        self.zero_probs: NDArray[np.float64] | None = None
        self.verbose = verbose

    def fit(self, data: NDArray[np.str_] | Sequence[str]) -> None:
        self.vocab = CharVocab.from_data(data)
        self.default_probs = np.hstack(
            [np.ones(len(self.vocab) - 4), np.array([0.0, 1.0, 0.0, 0.0])]
        )
        self.zero_probs = np.zeros(len(self.vocab))
        if self.verbose:
            print("fitting...")
            data = tqdm(data, total=len(data))
        for line in data:
            t_line = tuple(self.vocab.string2ids(line, True, True))
            for i in range(len(t_line)):
                for shift in range(self.max_context_len):
                    if i + shift + 1 >= len(t_line):
                        break
                    context = t_line[i : i + shift + 1]
                    cid = t_line[i + shift + 1]
                    probs = self._dict.get(context, self.zero_probs.copy())
                    probs[cid] += 1.0
                    self._dict[context] = probs

    def fit_update(self, data: NDArray[np.str_] | Sequence[str]) -> None:
        if self.vocab is None or self.zero_probs is None:
            raise RuntimeError("Error: Fit the model before generating")

        iter_data = data
        if self.verbose:
            print("fitting...")
            iter_data = tqdm(data, total=len(data))
        for line in iter_data:
            t_line = tuple(self.vocab.string2ids(line, True, True))
            for i in range(len(t_line)):
                for shift in range(self.max_context_len):
                    if i + shift + 1 >= len(t_line):
                        break
                    context = t_line[i : i + shift + 1]
                    cid = t_line[i + shift + 1]
                    probs = self._dict.get(context, self.zero_probs.copy())
                    probs[cid] += 1.0
                    self._dict[context] = probs

    def generate_one(
        self, l_smooth: float = 0.01, context_len: int | None = None, max_len: int = 100
    ) -> str:
        if self.vocab is None or self.default_probs is None:
            raise RuntimeError("Error: Fit the model before generating")

        if context_len is None or context_len <= 0 or context_len > self.max_context_len:
            context_len = self.max_context_len

        res = [self.vocab.bos]

        while res[-1] != self.vocab.eos and len(res) < max_len:
            begin_index = max(len(res) - context_len, 0)
            context = tuple(res[begin_index:])
            while context not in self._dict:
                context = context[1:]
            probs = self._dict[context]
            smoothed = probs + self.default_probs * l_smooth
            normed = smoothed / smoothed.sum()
            next_symbol = np.random.choice(len(self.vocab), p=normed)
            res.append(next_symbol)

        return self.vocab.ids2string(res)

    def nll(self, smiles: str, l_smooth: float = 0.01, context_len: int | None = None) -> float:
        if self.vocab is None or self.default_probs is None:
            raise RuntimeError("Error: model is not trained")

        if context_len is None or context_len <= 0 or context_len > self.max_context_len:
            context_len = self.max_context_len

        tokens = tuple(self.vocab.string2ids(smiles, True, True))

        likelihood = 0.0
        for i in range(1, len(tokens)):
            begin_index = max(i - context_len, 0)
            context = tokens[begin_index:i]
            while context not in self._dict:
                context = context[1:]

            probs = self._dict[context] + self.default_probs
            normed = probs / probs.sum()
            prob = normed[tokens[i]]
            if prob == 0.0:
                return np.inf
            likelihood -= np.log(prob)

        return likelihood

    def generate(
        self,
        n: int,
        l_smooth: float = 0.01,
        context_len: int | None = None,
        max_len: int = 100,
    ) -> list[str]:
        generator = (self.generate_one(l_smooth, context_len, max_len) for i in range(n))
        if self.verbose:
            print("generating...")
            generator = tqdm(generator, total=n)
        return list(generator)

    def save(self, path: str) -> None:
        """
        Saves a model using pickle
        Arguments:
            path: path to .pkl file for saving
        """
        if self.vocab is None:
            raise RuntimeError("Can't save empty model. Fit the model first")
        data = {
            "_dict": self._dict,
            "vocab": self.vocab,
            "default_probs": self.default_probs,
            "zero_probs": self.zero_probs,
            "max_context_len": self.max_context_len,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> Self:
        """
        Loads saved model
        Arguments:
            path: path to saved .pkl file
        Returns:
            Loaded NGramGenerator
        """
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        model = cls()
        model._dict = data["_dict"]
        model.vocab = data["vocab"]
        model.default_probs = data["default_probs"]
        model.zero_probs = data["zero_probs"]
        model.max_context_len = data["max_context_len"]

        return model


def reproduce(
    seed: int,
    samples_path: str | None = None,
    metrics_path: str | None = None,
    n_jobs: Pool | int = 1,
    device: str = "cpu",
    verbose: bool = False,
    samples: int = 30000,
) -> tuple[list[str], dict[str, float]]:
    data = moses.get_dataset("train")
    model = NGram(10, verbose=verbose)
    model.fit(data)
    np.random.seed(seed)
    smiles = model.generate(samples, l_smooth=0.01)
    metrics = moses.get_all_metrics(smiles, n_jobs=n_jobs, device=device)

    if samples_path is not None:
        with open(samples_path, "w") as out:
            out.write("SMILES\n")
            for s in smiles:
                out.write(s + "\n")

    if metrics_path is not None:
        with open(metrics_path, "w") as out:
            for key, value in metrics.items():
                out.write("%s,%f\n" % (key, value))

    return smiles, metrics
