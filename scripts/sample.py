import argparse
import sys

import pandas as pd
import torch
from rdkit import RDLogger
from tqdm.auto import tqdm

from moses.models_storage import ModelsStorage
from moses.script_utils import SampleConfig, add_sample_args, set_seed

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # type: ignore[no-untyped-call]

MODELS = ModelsStorage()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Models sampler script", description="available models"
    )
    for model in MODELS.get_model_names():
        add_sample_args(subparsers.add_parser(model))
    return parser


def main(model: str, config: SampleConfig) -> None:
    set_seed(config.seed)
    device = torch.device(config.device)

    # For CUDNN to work properly:
    if device.type.startswith("cuda"):
        torch.cuda.set_device(device.index or 0)

    model_config = torch.load(config.config_load)
    model_vocab = torch.load(config.vocab_load)
    model_state = torch.load(config.model_load)

    model_cls = MODELS.get_model_class(model)(model_vocab, model_config)  # type: ignore[call-arg]
    model_cls.load_state_dict(model_state)
    model_cls = model_cls.to(device)
    model_cls.eval()

    samples_ls: list[str] = []
    n = config.n_samples
    with tqdm(total=config.n_samples, desc="Generating samples") as T:
        while n > 0:
            current_samples = model_cls.sample(min(n, config.n_batch), config.max_len)
            samples_ls.extend(current_samples)

            n -= len(current_samples)
            T.update(len(current_samples))

    samples = pd.DataFrame(samples_ls, columns=["SMILES"])
    samples.to_csv(config.gen_save, index=False)


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_args(namespace=SampleConfig())
    model = sys.argv[1]
    main(model, config)
