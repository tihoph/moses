from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from moses.aae import AAE, AAETrainer, aae_parser
from moses.char_rnn import CharRNN, CharRNNTrainer, char_rnn_parser
from moses.latentgan import LatentGAN, LatentGANTrainer, latentGAN_parser
from moses.organ import ORGAN, ORGANTrainer, organ_parser
from moses.vae import VAE, VAETrainer, vae_parser

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    from torch import nn

    from moses.interfaces import MosesTrainer


class StoredModel(TypedDict):
    class_: type[nn.Module]
    trainer: type[MosesTrainer]
    parser: Callable[[argparse.ArgumentParser | None], argparse.ArgumentParser]


class ModelsStorage:
    def __init__(self) -> None:
        self._models: dict[str, StoredModel] = {}
        self.add_model("aae", AAE, AAETrainer, aae_parser)
        self.add_model("char_rnn", CharRNN, CharRNNTrainer, char_rnn_parser)
        self.add_model("vae", VAE, VAETrainer, vae_parser)
        self.add_model("organ", ORGAN, ORGANTrainer, organ_parser)
        self.add_model("latentgan", LatentGAN, LatentGANTrainer, latentGAN_parser)

    def add_model(
        self,
        name: str,
        class_: type[nn.Module],
        trainer_: type[MosesTrainer],
        parser_: Callable[[argparse.ArgumentParser | None], argparse.ArgumentParser],
    ) -> None:
        self._models[name] = {"class_": class_, "trainer": trainer_, "parser": parser_}

    def get_model_names(self) -> list[str]:
        return list(self._models.keys())

    def get_model_trainer(self, name: str) -> type[MosesTrainer]:
        return self._models[name]["trainer"]

    def get_model_class(self, name: str) -> type[nn.Module]:
        return self._models[name]["class_"]

    def get_model_train_parser(
        self, name: str
    ) -> Callable[[argparse.ArgumentParser | None], argparse.ArgumentParser]:
        return self._models[name]["parser"]
