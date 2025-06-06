import argparse
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance

from moses.dataset import get_dataset
from moses.metrics import QED, SA, logP, weight
from moses.metrics.utils import get_mol, mapper  # type: ignore[attr-defined]
from moses.utils import disable_rdkit_log


class PlotConfig(argparse.Namespace):
    config: str = "config.csv"
    n_jobs: int = 1
    img_folder: str = "images/"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Prepares distribution plots for weight, logP, SA, and QED\n")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=PlotConfig.config,
        help="Path to the config csv with `name` and `path` columns. "
        "`name` is a model name, and "
        "`path` is a path to generated samples`",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=PlotConfig.n_jobs, help="number of processes to use"
    )
    parser.add_argument(
        "--img_folder", type=str, default=PlotConfig.img_folder, help="Store images in this folder"
    )
    return parser


if __name__ == "__main__":
    disable_rdkit_log()
    parser = get_parser()
    config = parser.parse_args(namespace=PlotConfig())

    os.makedirs(config.img_folder, exist_ok=True)

    generated = OrderedDict({"MOSES": pd.DataFrame({"SMILES": get_dataset("test")})})
    models = pd.read_csv(config.config)
    for path, name in zip(models["path"], models["name"]):
        generated[name] = pd.read_csv(path)

    metrics = {"weight": weight, "logP": logP, "SA": SA, "QED": QED}

    for s in generated.values():
        s["ROMol"] = mapper(config.n_jobs)(get_mol, s["SMILES"])

    distributions: dict[str, dict[str, list[float]]] = {}
    for metric_name, metric_fn in metrics.items():
        distributions[metric_name] = {}
        for _set, _molecules in generated.items():
            distributions[metric_name][_set] = mapper(config.n_jobs)(
                metric_fn, _molecules["ROMol"].dropna().values
            )

    for metric_i, metric_name in enumerate(metrics):
        for model, d in distributions[metric_name].items():
            dist = wasserstein_distance(distributions[metric_name]["MOSES"], d)
            sns.distplot(
                d,
                hist=False,
                kde=True,
                kde_kws={"shade": True, "linewidth": 3},
                label="{0} ({1:0.2g})".format(model, dist),
            )
        plt.title(metric_name, fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.img_folder, metric_name + ".pdf"))
        plt.savefig(os.path.join(config.img_folder, metric_name + ".png"), dpi=250)
        plt.close()
