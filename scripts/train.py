import argparse
import sys

import torch
from rdkit import RDLogger

from moses.dataset import get_dataset
from moses.models_storage import ModelsStorage
from moses.script_utils import TrainConfig, add_train_args, read_smiles_csv, set_seed

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # type: ignore[no-untyped-call]

MODELS = ModelsStorage()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Models trainer script", description="available models"
    )
    for model in MODELS.get_model_names():
        add_train_args(MODELS.get_model_train_parser(model)(subparsers.add_parser(model)))
    return parser


def main(model: str, config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device(config.device)

    if config.config_save is not None:
        torch.save(config, config.config_save)

    # For CUDNN to work properly
    if device.type.startswith("cuda"):
        torch.cuda.set_device(device.index or 0)

    train_data = (
        get_dataset("train") if config.train_load is None else read_smiles_csv(config.train_load)
    )

    val_data = get_dataset("test") if config.val_load is None else read_smiles_csv(config.val_load)

    trainer = MODELS.get_model_trainer(model)(config)  # type: ignore[call-arg]

    vocab = (
        torch.load(config.vocab_load)
        if config.vocab_load is not None
        else trainer.get_vocabulary(train_data)
    )

    if config.vocab_save is not None:
        torch.save(vocab, config.vocab_save)

    model_cls = MODELS.get_model_class(model)(vocab, config).to(device)  # type: ignore[call-arg]
    trainer.fit(model_cls, train_data, val_data)

    model_cls = model_cls.to("cpu")
    torch.save(model_cls.state_dict(), config.model_save)


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_args(namespace=TrainConfig())
    model = sys.argv[1]
    main(model, config)
