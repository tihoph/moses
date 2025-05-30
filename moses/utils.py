from __future__ import annotations

import random
from collections import UserList, defaultdict
from multiprocessing.pool import Pool
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from rdkit import Chem, rdBase
from torch.nn.utils.rnn import PackedSequence
from typing_extensions import ParamSpec, Self, override

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

GruOutT = tuple[PackedSequence, torch.Tensor]
LstmOutT = tuple[PackedSequence, tuple[torch.Tensor, torch.Tensor]]


# https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
def set_torch_seed_to_all_gens(_: object) -> None:
    seed = torch.initial_seed() % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)


class SpecialTokensProto(Protocol):
    bos: str
    eos: str
    pad: str
    unk: str


class SpecialTokens:
    bos = "<bos>"
    eos = "<eos>"
    pad = "<pad>"
    unk = "<unk>"


class CharVocab:
    @classmethod
    def from_data(
        cls,
        data: NDArray[np.str_] | Sequence[str],
        ss: type[SpecialTokensProto] = SpecialTokens,
    ) -> Self:
        chars: set[str] = set()
        for string in data:
            chars.update(string)

        return cls(chars, ss)

    def __init__(
        self, chars: Collection[str], ss: type[SpecialTokensProto] = SpecialTokens
    ) -> None:
        if (ss.bos in chars) or (ss.eos in chars) or (ss.pad in chars) or (ss.unk in chars):
            raise ValueError("SpecialTokens in chars")

        all_syms = [*sorted(chars), ss.bos, ss.eos, ss.pad, ss.unk]

        self.ss = ss
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = dict(enumerate(all_syms))

    def __len__(self) -> int:
        return len(self.c2i)

    @property
    def bos(self) -> int:
        return self.c2i[self.ss.bos]

    @property
    def eos(self) -> int:
        return self.c2i[self.ss.eos]

    @property
    def pad(self) -> int:
        return self.c2i[self.ss.pad]

    @property
    def unk(self) -> int:
        return self.c2i[self.ss.unk]

    def char2id(self, char: str) -> int:
        if char not in self.c2i:
            return self.unk

        return self.c2i[char]

    def id2char(self, id: int) -> str:  # noqa: A002
        if id not in self.i2c:
            return self.ss.unk

        return self.i2c[id]

    def string2ids(self, string: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.char2id(c) for c in string]

        if add_bos:
            ids = [self.bos, *ids]
        if add_eos:
            ids = [*ids, self.eos]

        return ids

    def ids2string(self, ids: Sequence[int], rem_bos: bool = True, rem_eos: bool = True) -> str:
        if len(ids) == 0:
            return ""
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:-1]

        return "".join([self.id2char(id) for id in ids])  # noqa: A001


class OneHotVocab(CharVocab):
    def __init__(
        self, chars: Collection[str], ss: type[SpecialTokensProto] = SpecialTokens
    ) -> None:
        super().__init__(chars, ss)
        self.vectors = torch.eye(len(self.c2i))


def mapper(
    n_jobs: Pool | int,
) -> Callable[[Callable[[T], R], Iterable[T]], list[R]]:
    """
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    """
    if n_jobs == 1:

        def _mapper(func: Callable[[T], R], iterable: Iterable[T]) -> list[R]:
            return list(map(func, iterable))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(func: Callable[[T], R], iterable: Iterable[T]) -> list[R]:
            try:
                result = pool.map(func, iterable)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map


class Logger(UserList[dict[str, Any]]):
    def __init__(self, data: Sequence[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self.sdata: defaultdict[str, list[Any]] = defaultdict(list)
        for step in data or []:
            self.append(step)

    @overload  # type: ignore[override]
    def __getitem__(self, key: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, key: slice) -> Logger: ...

    @overload
    def __getitem__(self, key: str) -> list[Any]: ...

    @override
    def __getitem__(self, key: int | slice | str) -> dict[str, Any] | list[Any] | Logger:
        if isinstance(key, int):
            return self.data[key]
        if isinstance(key, slice):
            return Logger(self.data[key])
        ldata = self.sdata[key]
        if isinstance(ldata[0], dict):
            return Logger(ldata)
        return ldata

    @override
    def append(self, step_dict: dict[str, Any]) -> None:
        super().append(step_dict)
        for k, v in step_dict.items():
            self.sdata[k].append(v)

    def save(self, path: str) -> None:
        df = pd.DataFrame(list(self))
        df.to_csv(path, index=None)


class LogPlotter:
    def __init__(self, log: Logger) -> None:
        self.log = log

    def line(self, ax: Axes, name: str) -> None:
        if isinstance(self.log[0][name], dict):
            for k in self.log[0][name]:
                ax.plot(self.log[name][k], label=k)
            ax.legend()
        else:
            ax.plot(self.log[name])

        ax.set_ylabel("value")
        ax.set_xlabel("epoch")
        ax.set_title(name)

    def grid(self, names: Sequence[str], size: int = 7) -> None:
        _, axs = plt.subplots(
            nrows=len(names) // 2, ncols=2, figsize=(size * 2, size * (len(names) // 2))
        )

        for ax, name in zip(axs.flatten(), names):
            self.line(ax, name)


class CircularBuffer:
    def __init__(self, size: int) -> None:
        self.max_size = size
        self.data = np.zeros(self.max_size)
        self.size = 0
        self.pointer = -1

    def add(self, element: float) -> float:
        self.size = min(self.size + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size
        self.data[self.pointer] = element
        return element

    def last(self) -> float:
        assert self.pointer != -1, "Can't get an element from an empty buffer!"
        return self.data[self.pointer]  # type: ignore[no-any-return]

    def mean(self) -> float:
        if self.size > 0:
            return self.data[: self.size].mean()  # type: ignore[no-any-return]
        return 0.0


def disable_rdkit_log() -> None:
    rdBase.DisableLog("rdApp.*")


def enable_rdkit_log() -> None:
    rdBase.EnableLog("rdApp.*")


def get_mol(smiles_or_mol: str | Chem.Mol) -> Chem.Mol | None:
    """
    Loads SMILES/molecule into RDKit's object
    """
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol: Chem.Mol | None = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


class StringDataset:
    def __init__(self, vocab: CharVocab, data: Sequence[str]) -> None:
        """
        Creates a convenient Dataset with SMILES tokinization

        Arguments:
            vocab: CharVocab instance for tokenization
            data (list): SMILES strings for the dataset
        """
        self.vocab = vocab
        self.tokens = [vocab.string2ids(s) for s in data]
        self.data = data
        self.bos = vocab.bos
        self.eos = vocab.eos

    def __len__(self) -> int:
        """
        Computes a number of objects in the dataset
        """
        return len(self.tokens)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Prepares torch tensors with a given SMILES.

        Arguments:
            index (int): index of SMILES in the original dataset

        Returns:
            A tuple (with_bos, with_eos, smiles), where
            * with_bos is a torch.long tensor of SMILES tokens with
                BOS (beginning of a sentence) token
            * with_eos is a torch.long tensor of SMILES tokens with
                EOS (end of a sentence) token
            * smiles is an original SMILES from the dataset
        """
        tokens = self.tokens[index]
        with_bos = torch.tensor([self.bos, *tokens], dtype=torch.long)
        with_eos = torch.tensor([*tokens, self.eos], dtype=torch.long)
        return with_bos, with_eos, self.data[index]

    def default_collate(
        self,
        batch: Sequence[tuple[torch.Tensor, torch.Tensor, str]],
        return_data: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, list[int], NDArray[np.str_]]
        | tuple[torch.Tensor, torch.Tensor, list[int]]
    ):
        """
        Simple collate function for SMILES dataset. Joins a
        batch of objects from StringDataset into a batch

        Arguments:
            batch: list of objects from StringDataset
            pad: padding symbol, usually equals to vocab.pad
            return_data: if True, will return SMILES used in a batch

        Returns:
            with_bos, with_eos, lengths [, data] where
            * with_bos: padded sequence with BOS in the beginning
            * with_eos: padded sequence with EOS in the end
            * lengths: array with SMILES lengths in the batch
            * data: SMILES in the batch

        Note: output batch is sorted with respect to SMILES lengths in
            decreasing order, since this is a default format for torch
            RNN implementations
        """
        with_bos, with_eos, data = list(zip(*batch))
        lengths = [len(x) for x in with_bos]
        order: Sequence[int] = np.argsort(lengths)[::-1]  # type: ignore[assignment]
        with_bos_ls: list[torch.Tensor] = [with_bos[i] for i in order]
        with_eos_ls: list[torch.Tensor] = [with_eos[i] for i in order]
        lengths = [lengths[i] for i in order]
        with_bos_padded = torch.nn.utils.rnn.pad_sequence(with_bos_ls, padding_value=self.vocab.pad)
        with_eos_padded = torch.nn.utils.rnn.pad_sequence(with_eos_ls, padding_value=self.vocab.pad)
        if return_data:
            data_arr: NDArray[np.str_] = np.array(data)[order]
            return with_bos_padded, with_eos_padded, lengths, data_arr
        return with_bos_padded, with_eos_padded, lengths


def batch_to_device(
    batch: Sequence[torch.Tensor | T], device: torch.device
) -> list[torch.Tensor | T]:
    return [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
